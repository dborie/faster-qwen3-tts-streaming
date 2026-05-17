"""
High-level streaming-text-input session.

Wraps the model + the streaming decode loop + a worker thread.
Caller pattern:

    session = StreamingSession(model, mode="custom", speaker="aiden")
    session.start()
    session.feed_text("The quick brown fox ")
    for chunk in session.audio_chunks(yield_until_done=False):
        speakers.write(chunk)        # play as it arrives
    session.feed_text("jumps over the lazy dog.")
    session.complete()
    for chunk in session.audio_chunks(yield_until_done=True):
        speakers.write(chunk)        # remaining audio + EOS
    session.close()

Or plumb via async iteration / queue if integrating with FastAPI.

The decode loop runs in a daemon thread. Caller-side text writes go
through `_buffer_lock` so they don't race with the decode loop's reads.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Iterator, List, Optional

import numpy as np
import torch

from .decode_loop import streaming_fast_generate, text_to_trailing_embed


# Reasonable upper bound on text-token positions we'll ever need in one
# session. Sentences past this would need a bigger buffer; we just fail
# loudly rather than truncating silently.
MAX_TEXT_POSITIONS = 2048


class _SessionClosedExit(BaseException):
    """Raised from on_step when close() is called mid-decode so the
    streaming generator unwinds promptly instead of running to
    max_new_tokens. BaseException (not Exception) so internal
    except-Exception handlers in the decode loop can't swallow it."""


@dataclass
class SessionConfig:
    mode: str = "custom"                 # "clone" | "custom" | "design"
    # custom mode
    speaker: Optional[str] = None
    # design / custom mode style instruction
    instruct: Optional[str] = None
    # clone mode
    ref_audio: Optional[str] = None
    ref_text:  str = ""
    # Text baked into the prefix at session start. The model was
    # trained on prefixes containing real content text, so passing a
    # short first word or phrase here (and only streaming the rest)
    # gives noticeably more reliable output than seeding with " ".
    # Whatever value lands here is *not* re-fed via feed_text() - the
    # caller is responsible for not sending it twice.
    initial_text: str = " "
    language: str = "English"
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    do_sample: bool = True
    max_new_tokens: int = 4096
    min_new_tokens: int = 2

    def validate(self) -> None:
        if self.mode not in ("clone", "custom", "design"):
            raise ValueError(
                f"streaming session mode must be 'clone', 'custom', or 'design', not {self.mode!r}.")
        if self.mode == "custom" and not self.speaker:
            raise ValueError("custom mode requires speaker")
        if self.mode == "design" and not self.instruct:
            raise ValueError("design mode requires instruct")
        if self.mode == "clone" and not self.ref_audio:
            raise ValueError("clone mode requires ref_audio")


class StreamingSession:
    """
    Owns one in-flight streaming synthesis. Single-shot: once a session
    is closed, build a new one for the next utterance.
    """

    def __init__(self, model_wrapper, config: SessionConfig):
        config.validate()
        self.model_wrapper = model_wrapper
        self.config        = config
        self.sample_rate   = model_wrapper.sample_rate

        self._buffer_lock     = threading.Lock()
        self._audio_queue: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._fed_text_lock   = threading.Lock()
        self._pending_text: List[str] = []
        self._stream_complete = [False]
        self._fill_pointer    = [0]                  # mutable, shared with decode loop
        self._last_gen_step   = 0                    # for drain() outside on_step
        self._closed          = False
        self._worker: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None

    # ----- public API -----

    def start(self) -> None:
        """Spawn the decode worker. Call before feed_text()."""
        if self._worker is not None:
            raise RuntimeError("session already started")
        self._worker = threading.Thread(
            target=self._run, name="tts-streaming-worker", daemon=True)
        self._worker.start()

    def feed_text(self, text: str) -> None:
        """
        Append text to the decode stream. Tokenisation + embedding happens
        on the calling thread; the resulting embeddings are queued and the
        decode loop pulls them at its next on_step opportunity.
        """
        if not text:
            return
        if self._stream_complete[0]:
            raise RuntimeError("feed_text called after complete()")
        with self._fed_text_lock:
            self._pending_text.append(text)

    def complete(self) -> None:
        """Signal end-of-text. Decode finishes any queued text, then EOS-terminates."""
        self._stream_complete[0] = True

    def audio_chunks(self, yield_until_done: bool = True, timeout: Optional[float] = None
                     ) -> Iterator[bytes]:
        """
        Yield raw 16-bit little-endian mono PCM chunks at self.sample_rate
        as they're produced. None marks end-of-stream.

        With yield_until_done=False, returns whatever's currently queued
        and exits immediately; useful for tight loops that interleave
        feed_text() and audio collection. With yield_until_done=True,
        blocks until the worker emits the EOS sentinel.
        """
        if not yield_until_done:
            while True:
                try:
                    item = self._audio_queue.get_nowait()
                except queue.Empty:
                    return
                if item is None:
                    return
                yield item
            return
        while True:
            try:
                item = self._audio_queue.get(timeout=timeout)
            except queue.Empty:
                continue
            if item is None:
                return
            yield item

    def close(self) -> None:
        """Stop the worker, raise any accumulated error."""
        self._closed = True
        self._stream_complete[0] = True   # release the decode loop if it's waiting
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=10.0)
        if self._error is not None:
            raise self._error

    # ----- internals -----

    def _build_prefix(self):
        """
        Run the model's standard _prepare_generation logic with an empty
        seed text so we get the prefix tensors we need without committing
        to any text up front. Returns (talker_input_embeds, attention_mask,
        original_trailing, tts_pad_embed).
        """
        # Default seed is a single space (tokenizer-safe placeholder).
        # Callers in clone mode should override via SessionConfig.initial_text
        # because the model expects real content text in the prefix at
        # decode start - a placeholder leads to garbage on that
        # checkpoint.
        seed = self.config.initial_text or " "
        if self.config.mode == "custom":
            audio_arrays, sr = self._invoke_with_capture(
                lambda: self.model_wrapper.generate_custom_voice(
                    text=seed,
                    language=self.config.language,
                    speaker=self.config.speaker,
                    instruct=self.config.instruct or "",
                    non_streaming_mode=False,
                    max_new_tokens=1,        # we only want the prefix; immediate EOS is fine
                    min_new_tokens=1,
                ))
        elif self.config.mode == "design":
            audio_arrays, sr = self._invoke_with_capture(
                lambda: self.model_wrapper.generate_voice_design(
                    text=seed,
                    instruct=self.config.instruct,
                    language=self.config.language,
                    non_streaming_mode=False,
                    max_new_tokens=1,
                    min_new_tokens=1,
                ))
        else:                                # clone
            # Same trick as custom/design but using generate_voice_clone:
            # build the ICL prefix from the reference + a 1-char seed,
            # capture talker_input_embeds + trailing buffer. The trailing
            # buffer returned by generate_icl_prompt with
            # non_streaming_mode=False is the per-step text slot the
            # streaming patch mutates. Experimental: the model was
            # trained on prefixes containing real content text rather
            # than a placeholder, so behaviour with all-streamed text
            # is not guaranteed.
            audio_arrays, sr = self._invoke_with_capture(
                lambda: self.model_wrapper.generate_voice_clone(
                    text=seed,
                    language=self.config.language,
                    ref_audio=self.config.ref_audio,
                    ref_text=self.config.ref_text or "",
                    non_streaming_mode=False,
                    max_new_tokens=1,
                    min_new_tokens=1,
                ))

        captured = self._captured
        return (
            captured["talker_input_embeds"],
            captured["attention_mask"],
            captured["trailing_text_hidden"],
            captured["tts_pad_embed"],
        )

    def _invoke_with_capture(self, callable_):
        """
        Hijack faster_qwen3_tts.generate.fast_generate for one call so we
        can grab the prefix tensors before they're consumed. The actual
        generate call is short-circuited (1 token) since we only want the
        plumbing.
        """
        import faster_qwen3_tts.generate as fqt_gen
        original = fqt_gen.fast_generate
        captured = {}

        def capture(talker, talker_input_embeds, attention_mask,
                    trailing_text_hiddens, tts_pad_embed, config,
                    predictor_graph, talker_graph, **kwargs):
            captured["talker"]              = talker
            captured["talker_input_embeds"] = talker_input_embeds
            captured["attention_mask"]      = attention_mask
            captured["trailing_text_hidden"] = trailing_text_hiddens
            captured["tts_pad_embed"]       = tts_pad_embed
            captured["config"]              = config
            captured["predictor_graph"]     = predictor_graph
            captured["talker_graph"]        = talker_graph
            # Return one trivial codec frame so the caller doesn't crash.
            num_cb = config.num_code_groups
            stub = torch.zeros((1, num_cb), dtype=torch.long, device=trailing_text_hiddens.device)
            return stub, {"prefill_ms": 0, "decode_s": 0, "steps": 1, "ms_per_step": 0}

        fqt_gen.fast_generate = capture
        try:
            callable_()
        finally:
            fqt_gen.fast_generate = original

        self._captured = captured
        return [np.zeros(1, dtype=np.float32)], self.sample_rate

    def _run(self) -> None:
        try:
            (talker_input_embeds, attention_mask, original_trailing, tts_pad_embed) \
                = self._build_prefix()

            # Pre-allocate the streaming buffer at MAX_TEXT_POSITIONS, all pad.
            buffer = tts_pad_embed.expand(1, MAX_TEXT_POSITIONS, -1).clone()
            self._streaming_buffer = buffer

            # Run streaming decode. The on_step hook drains the pending-text
            # queue into the buffer at positions ahead of gen_step; that's
            # the actual streaming-text mutation point.
            captured = self._captured

            def drain():
                with self._fed_text_lock:
                    pending = self._pending_text
                    self._pending_text = []
                if not pending:
                    return
                joined = "".join(pending)
                emb = text_to_trailing_embed(self.model_wrapper, joined)
                t_len = emb.shape[1]
                with self._buffer_lock:
                    # Write at the model's current read position.
                    # on_step runs before inputs_embeds is built for
                    # the step, so the model picks up whatever we
                    # write here on the very same step.
                    nonlocal_gen = self._last_gen_step
                    start = max(self._fill_pointer[0], nonlocal_gen)
                    end   = start + t_len
                    if end > buffer.shape[1]:
                        self._stream_complete[0] = True
                        return
                    buffer[:, start:end, :] = emb
                    self._fill_pointer[0] = end

            def on_step(step_idx, gen_step, trailing_buf, fill_ptr, codec_so_far):
                if self._closed:
                    self._stream_complete[0] = True
                    raise _SessionClosedExit()
                self._last_gen_step = gen_step
                drain()
                # Flow control: if the model has caught up to where real
                # text ends, pause the decode loop here until more text
                # arrives or the stream is signalled complete. The model
                # generating off pad embeddings produces nonsense audio,
                # not silence, so just blocking the Python loop is the
                # right behaviour.
                paused_once = False
                while (gen_step >= fill_ptr[0]
                       and not self._stream_complete[0]
                       and not self._closed):
                    # First time we hit the pause point, flush whatever
                    # codec has been produced so far so the user hears
                    # the partial output instead of waiting for the
                    # next chunk-size boundary while the loop sits idle.
                    if not paused_once and len(codec_so_far) > 0:
                        emit_new_audio(torch.stack(codec_so_far))
                    paused_once = True
                    time.sleep(0.02)
                    drain()

            # Incremental codec -> waveform: every codec_chunk_size frames
            # the decode loop hands us the cumulative codec; we run the
            # vocoder and emit just the new audio tail. Re-decoding the
            # whole codec each chunk is O(N^2) on the vocoder side but
            # avoids boundary artifacts; sliding-window context would be
            # the next optimisation.
            inference_wrapper = self.model_wrapper.model
            emitted_samples = [0]

            def emit_new_audio(codec_so_far):
                if self._closed:
                    return
                try:
                    wavs, _sr = inference_wrapper.model.speech_tokenizer.decode(
                        [{"audio_codes": codec_so_far}])
                except Exception:
                    return
                # speech_tokenizer.decode returns torch tensors in some
                # paths and numpy arrays in others; handle both.
                raw = wavs[0]
                if hasattr(raw, "cpu"):
                    audio = raw.cpu().numpy().astype(np.float32)
                else:
                    audio = np.asarray(raw, dtype=np.float32)
                new_audio = audio[emitted_samples[0]:]
                if new_audio.size == 0:
                    return
                pcm16 = np.clip(new_audio * 32768, -32768, 32767).astype(np.int16)
                self._audio_queue.put(pcm16.tobytes())
                emitted_samples[0] = len(audio)

            codec_ids, timing = streaming_fast_generate(
                talker=captured["talker"],
                talker_input_embeds=talker_input_embeds,
                attention_mask=attention_mask,
                trailing_text_hidden=buffer,
                text_fill_pointer=self._fill_pointer,
                stream_complete=self._stream_complete,
                tts_pad_embed=tts_pad_embed,
                config=captured["config"],
                predictor_graph=captured["predictor_graph"],
                talker_graph=captured["talker_graph"],
                on_step=on_step,
                on_codec_chunk=emit_new_audio,
                # ~330ms chunks at 12Hz codec: vocode often enough that
                # a single pushed word produces audible output without
                # exploding the O(N^2) re-decode cost.
                codec_chunk_size=4,
                max_new_tokens=self.config.max_new_tokens,
                min_new_tokens=self.config.min_new_tokens,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample,
                repetition_penalty=self.config.repetition_penalty,
            )

            # Final tail: any audio past the last incremental emit.
            if codec_ids is not None:
                emit_new_audio(codec_ids)

        except _SessionClosedExit:
            # close() was called mid-decode. Clean stop, not an error.
            pass
        except BaseException as exc:
            self._error = exc
        finally:
            self._audio_queue.put(None)         # EOS sentinel
