#!/usr/bin/env python3
"""
Qwen3-TTS demo server. Five engine modes side by side:

  normal              : original qwen_tts decode loop (no CUDA graphs).
                        On Base, generate_voice_clone_streaming with
                        parity_mode=True. On CustomVoice / VoiceDesign
                        the upstream library doesn't expose parity, so
                        we monkey-patch fast_generate_streaming for
                        the duration of the request.
  faster              : CUDA-graph decode loop, response held until
                        all PCM is produced.
  faster_stream       : same decode loop, response flushed per chunk
                        as the model emits.
  input_stream        : streaming-text-input via the /stream WS;
                        SmartSession in pass-through (per-word feed).
  smart_input_stream  : same WS endpoint with SmartSession's adaptive
                        batching enabled.

Model + (Base | CustomVoice | VoiceDesign) variant is picked per
request by the client. A reference WAV + transcript in 'source voice/'
is required for Base mode and can be uploaded via the UI.

Run:
    pip install "faster-qwen3-tts-streaming[demo]"
    python server.py
"""
import asyncio
import gc
import io
import json
import os
import queue
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Optional

# Make the parent dir importable so 'from faster_qwen3_tts_streaming
# import ...' finds the package sitting next to this demo dir.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict


HERE             = Path(__file__).parent
SOURCE_VOICE_DIR = HERE / "source voice"
STATIC_DIR       = HERE / "static"

DEFAULT_SIZE = os.environ.get("QWEN_TTS_DEFAULT_SIZE", "1.7B")
DEFAULT_TYPE = os.environ.get("QWEN_TTS_DEFAULT_TYPE", "Base")

VALID_SIZES = ("0.6B", "1.7B")
VALID_TYPES = ("Base", "CustomVoice", "VoiceDesign")

app = FastAPI(title="Qwen3-TTS demo")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Cache of loaded model checkpoints keyed on (size, type). Each holds
# ~3 GB of VRAM; we don't evict, so users who churn through many
# combinations will accumulate state.
_models: dict = {}
_models_lock = threading.Lock()

# Held while normal mode for CV/VD temporarily monkey-patches
# faster_qwen3_tts.streaming.fast_generate_streaming. Stops a second
# concurrent request from seeing the patched function unexpectedly.
_parity_patch_lock = threading.Lock()


def _model_id(size: str, type_: str) -> str:
    return f"Qwen/Qwen3-TTS-12Hz-{size}-{type_}"


def _get_model(size: str, type_: str):
    if size not in VALID_SIZES:
        raise HTTPException(400, f"unknown model size: {size!r}")
    if type_ not in VALID_TYPES:
        raise HTTPException(400, f"unknown model type: {type_!r}")
    key = (size, type_)
    with _models_lock:
        cached = _models.get(key)
    if cached is not None:
        return cached
    from faster_qwen3_tts import FasterQwen3TTS
    mid = _model_id(size, type_)
    print(f"[demo] loading {mid} ...", flush=True)
    model = FasterQwen3TTS.from_pretrained(
        mid, device="cuda", dtype=torch.bfloat16,
    )
    print(f"[demo] {mid} ready, sample_rate={model.sample_rate}", flush=True)
    with _models_lock:
        _models[key] = model
    return model


def _reference():
    """Return (audio_path, transcript) for the currently-set reference.

    Raises HTTPException(400) if the user hasn't uploaded one yet -
    clone-mode synthesis needs it, custom / design don't.
    """
    SOURCE_VOICE_DIR.mkdir(parents=True, exist_ok=True)
    wavs = sorted(SOURCE_VOICE_DIR.glob("*.wav"))
    if not wavs:
        raise HTTPException(
            400,
            "No reference voice set. Upload a WAV + transcript via the "
            "'Reference voice' panel in the demo UI before using clone "
            "(Base) mode.",
        )
    ref_audio = wavs[0]
    ref_text_file = ref_audio.with_suffix(".txt")
    ref_text = ""
    if ref_text_file.exists():
        ref_text = ref_text_file.read_text(encoding="utf-8").strip()
    return str(ref_audio), ref_text


def _streaming_wav_header(sample_rate: int) -> bytes:
    # 44-byte header with 0xFFFFFFFF sentinels (unknown length).
    n_channels, bits = 1, 16
    byte_rate   = sample_rate * n_channels * bits // 8
    block_align = n_channels * bits // 8
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 0xFFFFFFFF))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, n_channels, sample_rate, byte_rate, block_align, bits))
    buf.write(b"data")
    buf.write(struct.pack("<I", 0xFFFFFFFF))
    return buf.getvalue()


def _to_pcm16(pcm) -> bytes:
    return np.clip(pcm * 32768, -32768, 32767).astype(np.int16).tobytes()


def _resolve_speaker(model, requested: Optional[str]) -> str:
    """Pick a CustomVoice speaker. Falls back to the first supported
    name on the loaded model, or 'aiden' if the model doesn't expose
    a roster."""
    if requested:
        return requested
    try:
        names = list(model.model.get_supported_speakers() or [])
    except Exception:
        names = []
    return names[0] if names else "aiden"


class SynthRequest(BaseModel):
    # Pydantic V2 reserves the 'model_' prefix for its own internals
    # (model_dump, model_config, ...). Open the namespace so our
    # model_size / model_type fields aren't flagged.
    model_config = ConfigDict(protected_namespaces=())

    text:               str
    mode:               str   = "faster"   # normal | faster | faster_stream
    language:           str   = "English"
    temperature:        float = 0.9
    top_k:              int   = 50
    repetition_penalty: float = 1.05
    seed:               int   = -1
    model_size:         Optional[str] = None
    model_type:         Optional[str] = None
    speaker:            Optional[str] = None
    instruct:           Optional[str] = None


class WarmupRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_size: Optional[str] = None
    model_type: Optional[str] = None


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/reference")
async def reference_info():
    SOURCE_VOICE_DIR.mkdir(parents=True, exist_ok=True)
    wavs = sorted(SOURCE_VOICE_DIR.glob("*.wav"))
    if not wavs:
        return {"file": None, "text": ""}
    ref_audio = wavs[0]
    ref_text_file = ref_audio.with_suffix(".txt")
    ref_text = ref_text_file.read_text(encoding="utf-8").strip() if ref_text_file.exists() else ""
    return {"file": Path(ref_audio).name, "text": ref_text}


@app.post("/reference")
async def reference_set(audio: UploadFile = File(...), text: str = Form("")):
    """Replace the current reference voice with an uploaded WAV +
    transcript. Existing WAV/TXT pairs in 'source voice/' are removed
    so _reference() always picks the new one."""
    if not audio.filename or not audio.filename.lower().endswith(".wav"):
        raise HTTPException(400, "expected a .wav upload")
    SOURCE_VOICE_DIR.mkdir(parents=True, exist_ok=True)
    for old in list(SOURCE_VOICE_DIR.glob("*.wav")) + list(SOURCE_VOICE_DIR.glob("*.txt")):
        try: old.unlink()
        except OSError: pass
    target_wav = SOURCE_VOICE_DIR / "reference.wav"
    target_txt = SOURCE_VOICE_DIR / "reference.txt"
    data = await audio.read()
    target_wav.write_bytes(data)
    target_txt.write_text(text.strip(), encoding="utf-8")
    return {"file": target_wav.name, "text": text.strip(), "bytes": len(data)}


@app.get("/status")
async def status():
    return {
        "loaded": [{"size": s, "type": t} for (s, t) in sorted(_models.keys())],
        "default_size": DEFAULT_SIZE,
        "default_type": DEFAULT_TYPE,
    }


@app.get("/speakers")
async def speakers(size: Optional[str] = None, type: Optional[str] = None):
    """Built-in speaker names for a CustomVoice checkpoint."""
    size = size or DEFAULT_SIZE
    type_ = type or "CustomVoice"
    if type_ != "CustomVoice":
        return {"speakers": []}
    model = _get_model(size, type_)
    try:
        names = list(model.model.get_supported_speakers() or [])
    except Exception:
        names = []
    return {"size": size, "type": type_, "speakers": names}


class UnloadRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_size: str
    model_type: str


@app.post("/unload")
async def unload(req: UnloadRequest):
    """Drop a cached model and free its VRAM."""
    key = (req.model_size, req.model_type)
    with _models_lock:
        model = _models.pop(key, None)
    if model is None:
        return {"unloaded": False, "model_size": req.model_size, "model_type": req.model_type}
    del model
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return {"unloaded": True, "model_size": req.model_size, "model_type": req.model_type}


def _call_streaming_for_warmup(model, model_type: str, ref_audio: str, ref_text: str):
    """Tiny per-mode generate call to capture CUDA graphs."""
    if model_type == "Base":
        return model.generate_voice_clone_streaming(
            text="ok", language="English",
            ref_audio=ref_audio, ref_text=ref_text,
            parity_mode=False, max_new_tokens=8, min_new_tokens=2,
        )
    if model_type == "CustomVoice":
        return model.generate_custom_voice_streaming(
            text="ok", language="English", speaker=_resolve_speaker(model, None),
            max_new_tokens=8, min_new_tokens=2,
        )
    # VoiceDesign
    return model.generate_voice_design_streaming(
        text="ok", language="English",
        instruct="neutral voice",
        max_new_tokens=8, min_new_tokens=2,
    )


@app.post("/warmup")
async def warmup(req: WarmupRequest):
    """
    Force the slow stuff to happen before the first real synthesize:
        1. from_pretrained()              (~30 s on cold disk)
        2. first generate call captures   (~1-2 s) CUDA graphs.
    Idempotent per (size, type) pair.
    """
    size = req.model_size or DEFAULT_SIZE
    type_ = req.model_type or DEFAULT_TYPE
    load_t0 = time.perf_counter()
    model = _get_model(size, type_)
    load_ms = int((time.perf_counter() - load_t0) * 1000)

    # Only Base needs the reference voice for its warmup call;
    # CustomVoice + VoiceDesign warm up off built-in speaker / instruct.
    # If Base is warmed before a reference has been uploaded, do the
    # (slow) from_pretrained load above but skip the graph-capture
    # call - we'd have nothing to feed it. The first real synthesize
    # will pay the graph-capture cost then.
    ref_audio = ref_text = ""
    skipped_capture = False
    skip_reason = ""
    if type_ == "Base":
        try:
            ref_audio, ref_text = _reference()
        except HTTPException:
            skipped_capture = True
            skip_reason = "no reference uploaded - upload one to capture Base CUDA graphs"

    capture_ms = 0
    if not skipped_capture:
        capture_t0 = time.perf_counter()
        for _ in _call_streaming_for_warmup(model, type_, ref_audio, ref_text):
            pass
        capture_ms = int((time.perf_counter() - capture_t0) * 1000)

    return {
        "ok":              True,
        "model_size":      size,
        "model_type":      type_,
        "model_load_ms":   load_ms,
        "graph_warmup_ms": capture_ms,
        "skipped_capture": skipped_capture,
        "skip_reason":     skip_reason,
    }


@app.post("/synthesize")
async def synthesize(req: SynthRequest):
    if not req.text.strip():
        raise HTTPException(400, "text is empty")
    if req.mode not in ("normal", "faster", "faster_stream"):
        raise HTTPException(400, f"unknown mode: {req.mode}")

    if req.seed >= 0:
        torch.manual_seed(int(req.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(req.seed))

    size = req.model_size or DEFAULT_SIZE
    type_ = req.model_type or DEFAULT_TYPE
    model = _get_model(size, type_)
    # Only Base actually uses the reference WAV; CustomVoice + VoiceDesign
    # ignore ref_audio / ref_text, so don't 400 the request for missing
    # one when the user has selected those variants.
    if type_ == "Base":
        ref_audio, ref_text = _reference()
    else:
        ref_audio, ref_text = "", ""

    # Engine modes:
    #   normal        -> non-streaming generate methods (original
    #                    qwen_tts path, no CUDA graphs, slow).
    #   faster        -> streaming generate methods, response held until
    #                    all PCM is produced (CUDA graphs, buffered).
    #   faster_stream -> streaming generate methods, response flushed
    #                    per chunk as the model emits.
    flush_per_chunk = (req.mode == "faster_stream")

    def build_gen():
        """Streaming-method generator for the faster / faster_stream modes."""
        if type_ == "Base":
            return model.generate_voice_clone_streaming(
                text=req.text, language=req.language,
                ref_audio=ref_audio, ref_text=ref_text,
                temperature=req.temperature, top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
                parity_mode=False,
            )
        if type_ == "CustomVoice":
            return model.generate_custom_voice_streaming(
                text=req.text, language=req.language,
                speaker=_resolve_speaker(model, req.speaker),
                instruct=req.instruct or "",
                temperature=req.temperature, top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
            )
        # VoiceDesign
        return model.generate_voice_design_streaming(
            text=req.text, language=req.language,
            instruct=req.instruct or "Speak in a clear, neutral voice.",
            temperature=req.temperature, top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
        )

    def run_normal_buffered() -> bytes:
        """Normal mode: drive every variant through the parity (no-CUDA-graph)
        decode loop, so it's actually slower than 'faster'.

        Base uses the native parity_mode kwarg on its streaming method.
        CustomVoice / VoiceDesign don't expose parity_mode upstream, so
        we temporarily swap faster_qwen3_tts.streaming.fast_generate_streaming
        for a thin shim that calls parity_generate_streaming instead.
        The generate_*_streaming methods do `from .streaming import
        fast_generate_streaming` on every call, so module-level
        patching swaps the decode loop without touching anything else.
        """
        if type_ == "Base":
            stream = model.generate_voice_clone_streaming(
                text=req.text, language=req.language,
                ref_audio=ref_audio, ref_text=ref_text,
                temperature=req.temperature, top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
                parity_mode=True,
            )
            chunks = [header]
            for pcm_chunk, _sr, _t in stream:
                chunks.append(_to_pcm16(pcm_chunk))
            return b"".join(chunks)

        import faster_qwen3_tts.streaming as fqt_streaming
        original_fast = fqt_streaming.fast_generate_streaming
        parity_fn     = fqt_streaming.parity_generate_streaming

        def parity_shim(predictor_graph=None, talker_graph=None, **kwargs):
            return parity_fn(**kwargs)

        with _parity_patch_lock:
            fqt_streaming.fast_generate_streaming = parity_shim
            try:
                stream = build_gen()
                chunks = [header]
                for pcm_chunk, _sr, _t in stream:
                    chunks.append(_to_pcm16(pcm_chunk))
                return b"".join(chunks)
            finally:
                fqt_streaming.fast_generate_streaming = original_fast

    header = _streaming_wav_header(model.sample_rate)

    if req.mode == "normal":
        body = await asyncio.to_thread(run_normal_buffered)
        return Response(content=body, media_type="audio/wav")

    if not flush_per_chunk:
        # 'faster' (buffered): drain the streaming generator on a worker
        # thread, reply with one Response. No queue, no sync-generator-
        # in-StreamingResponse.
        def collect_all() -> bytes:
            chunks = [header]
            for pcm_chunk, _sr, _t in build_gen():
                chunks.append(_to_pcm16(pcm_chunk))
            return b"".join(chunks)

        body = await asyncio.to_thread(collect_all)
        return Response(content=body, media_type="audio/wav")

    # flush_per_chunk (faster_stream): real streaming. Producer thread
    # feeds a queue; the response iterator drains it.
    chunk_queue: queue.Queue = queue.Queue()
    SENTINEL = object()

    def producer():
        try:
            for pcm_chunk, _sr, _t in build_gen():
                chunk_queue.put(_to_pcm16(pcm_chunk))
        except Exception as exc:
            chunk_queue.put(("error", exc))
        finally:
            chunk_queue.put(SENTINEL)

    threading.Thread(target=producer, daemon=True).start()

    def streaming_body():
        yield header
        while True:
            item = chunk_queue.get()
            if item is SENTINEL:
                break
            if isinstance(item, tuple) and item[0] == "error":
                raise item[1]
            yield item

    return StreamingResponse(streaming_body(), media_type="audio/wav")


@app.websocket("/stream")
async def stream_endpoint(websocket: WebSocket):
    """
    Streaming-text-input WebSocket.

    Wire shape:
        client -> { "type": "init",      ...model + sampling args,
                                          smart_buffer?: bool,
                                          comfortable_lookahead_seconds?: float }
        client -> { "type": "text",      "content": "..." }       (zero or more)
        client -> { "type": "complete" } | { "type": "abort" }
        server -> { "type": "ready",     "sample_rate": ... }
        server -> binary PCM frames (16-bit LE mono)              (zero or more)
        server -> { "type": "done" } | { "type": "error", "message": "..." }

    Model + SessionConfig mode are picked from the init frame's
    model_size / model_type fields:

      Base        -> SmartSessionConfig(mode="clone", ref_audio, ref_text)
      CustomVoice -> SmartSessionConfig(mode="custom", speaker)
      VoiceDesign -> SmartSessionConfig(mode="design", instruct)

    smart_buffer enables SmartSession's adaptive batching - the
    wrapper holds incoming text and decides when to flush a batch to
    the inner StreamingSession based on its bootstrap and
    audio-buffer rules. With smart_buffer=False SmartSession is a
    pass-through, so each WS text frame is forwarded to the model
    immediately.

    The session is built right after the init frame with a
    placeholder initial_text; every subsequent WS text frame goes
    straight into session.feed_text() so SmartSession is the sole
    authority on cuts. Clone-mode streaming is experimental: ICL
    was trained on prefixes containing content text, and the
    placeholder seed may degrade voice quality - a library-level
    concern outside this demo's batching logic.
    """
    from faster_qwen3_tts_streaming import SmartSession, SmartSessionConfig

    await websocket.accept()
    session = None
    try:
        init_raw = await websocket.receive_text()
        init = json.loads(init_raw)
        if init.get("type") != "init":
            await websocket.send_text(json.dumps({"type": "error", "message": "first frame must be 'init'"}))
            return

        smart_buffer        = bool(init.get("smart_buffer", False))
        smart_min_initial   = int(init.get("smart_min_initial_words", 5))
        smart_max_initial   = int(init.get("smart_max_initial_words", 30))
        smart_safety_ms     = int(init.get("smart_safety_margin_ms", 300))

        # No server-side pre-accumulation, no prefix extraction.
        # SmartSession is the sole authority on cuts: every WS text
        # frame goes straight through session.feed_text() in the
        # reader loop below, and SmartSession decides when to forward
        # batched text into the inner StreamingSession.
        #
        # Note: the model is seeded with a placeholder initial_text
        # (" "). The original faster-qwen3-tts-streaming author noted
        # that clone-mode ICL was trained on prefixes containing real
        # content text; that's a library-level concern outside the
        # scope of this demo's batching logic.
        prefix_seed = " "

        # Pick model + SessionConfig mode from the UI's Model selectors.
        # Base -> clone (uses ref voice); CustomVoice -> custom (built-in
        # speaker); VoiceDesign -> design (free-form instruct only).
        size  = init.get("model_size") or DEFAULT_SIZE
        type_ = init.get("model_type") or DEFAULT_TYPE
        model = _get_model(size, type_)

        # SmartSessionConfig subclasses SessionConfig - smart_buffer=False
        # makes the wrapper a no-op pass-through, so this path stays
        # identical to plain StreamingSession behaviour unless the client
        # asks for the smart variant.
        if type_ == "Base":
            ref_audio, ref_text = _reference()
            cfg = SmartSessionConfig(
                mode="clone",
                ref_audio=ref_audio,
                ref_text=ref_text,
                initial_text=prefix_seed,
                language=init.get("language", "English"),
                instruct=init.get("instruct") or "",
                temperature=float(init.get("temperature", 0.9)),
                top_k=int(init.get("top_k", 50)),
                repetition_penalty=float(init.get("repetition_penalty", 1.05)),
                smart_buffer=smart_buffer,
                min_initial_words=smart_min_initial,
                max_initial_words=smart_max_initial,
                safety_margin_seconds=smart_safety_ms / 1000.0,
            )
        elif type_ == "CustomVoice":
            cfg = SmartSessionConfig(
                mode="custom",
                speaker=_resolve_speaker(model, init.get("speaker")),
                initial_text=prefix_seed,
                language=init.get("language", "English"),
                instruct=init.get("instruct") or "",
                temperature=float(init.get("temperature", 0.9)),
                top_k=int(init.get("top_k", 50)),
                repetition_penalty=float(init.get("repetition_penalty", 1.05)),
                smart_buffer=smart_buffer,
                min_initial_words=smart_min_initial,
                max_initial_words=smart_max_initial,
                safety_margin_seconds=smart_safety_ms / 1000.0,
            )
        else:  # VoiceDesign
            cfg = SmartSessionConfig(
                mode="design",
                instruct=init.get("instruct") or "Speak in a clear, neutral voice.",
                initial_text=prefix_seed,
                language=init.get("language", "English"),
                temperature=float(init.get("temperature", 0.9)),
                top_k=int(init.get("top_k", 50)),
                repetition_penalty=float(init.get("repetition_penalty", 1.05)),
                smart_buffer=smart_buffer,
                min_initial_words=smart_min_initial,
                max_initial_words=smart_max_initial,
                safety_margin_seconds=smart_safety_ms / 1000.0,
            )

        seed = init.get("seed", -1)
        if seed is not None and seed >= 0:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        session = SmartSession(model, cfg)
        session.start()

        # Serialise WS sends from the audio path and the on_flush
        # callback so they can't interleave bytes on the wire.
        ws_send_lock = asyncio.Lock()
        loop = asyncio.get_event_loop()

        async def _send_text(payload):
            async with ws_send_lock:
                try:
                    await websocket.send_text(json.dumps(payload))
                except (WebSocketDisconnect, RuntimeError):
                    pass

        async def _send_bytes(chunk):
            async with ws_send_lock:
                try:
                    await websocket.send_bytes(chunk)
                except (WebSocketDisconnect, RuntimeError):
                    pass

        # Report every SmartSession flush back to the client so the
        # UI can plot what got grouped into one feed_text call. Fires
        # synchronously from a worker thread, schedule the WS send on
        # the event loop.
        def _on_flush(text: str) -> None:
            asyncio.run_coroutine_threadsafe(
                _send_text({"type": "flush", "content": text}), loop)

        session.on_flush = _on_flush

        await _send_text({
            "type":          "ready",
            "sample_rate":   int(model.sample_rate),
            "initial_text":  prefix_seed,
            "initial_words": 0,
        })

        async def reader():
            try:
                while True:
                    msg_raw = await websocket.receive_text()
                    msg = json.loads(msg_raw)
                    kind = msg.get("type")
                    if kind == "text":
                        content = msg.get("content", "")
                        if content:
                            session.feed_text(content)
                    elif kind == "complete":
                        session.complete()
                        return
                    elif kind == "abort":
                        # Client hit Stop. Tear down the worker so the
                        # model stops generating - otherwise the writer
                        # keeps pulling chunks from the inner queue and
                        # the GPU keeps churning until natural EOS.
                        await loop.run_in_executor(None, session.close)
                        return
            except WebSocketDisconnect:
                # WS closed (eg client navigated away or hit Stop and
                # didn't get an abort frame in first). Same teardown.
                await loop.run_in_executor(None, session.close)
                return

        async def writer():
            chunks = session.audio_chunks(yield_until_done=True)
            iterator = iter(chunks)
            def pull_next():
                try:    return next(iterator)
                except StopIteration: return None
            while True:
                chunk = await loop.run_in_executor(None, pull_next)
                if chunk is None:
                    break
                await _send_bytes(chunk)
            # If the worker errored, surface it to the client instead
            # of sending the usual 'done' so the page doesn't
            # silently flip back to idle.
            worker_error = getattr(session, "_error", None)
            if worker_error is not None:
                import traceback as _tb
                tb_text = "".join(_tb.format_exception(
                    type(worker_error), worker_error, worker_error.__traceback__))
                print("[stream] worker error:\n" + tb_text, flush=True)
                await _send_text({
                    "type":    "error",
                    "message": f"{type(worker_error).__name__}: {worker_error}",
                })
            else:
                await _send_text({"type": "done"})

        await asyncio.gather(reader(), writer())

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass
    finally:
        if session is not None:
            try: session.close()
            except Exception: pass
        try: await websocket.close()
        except Exception: pass


if __name__ == "__main__":
    host = os.environ.get("DEMO_HOST", "127.0.0.1")
    port = int(os.environ.get("DEMO_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
