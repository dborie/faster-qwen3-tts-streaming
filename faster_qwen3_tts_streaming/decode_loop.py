"""
Streaming-aware fork of faster_qwen3_tts.generate.fast_generate.

Two changes vs upstream:

1.  The trailing_text_hidden buffer is treated as a MUTABLE shared
    tensor. The caller may write new text embeddings into positions
    ahead of `gen_step` between iterations. Each step reads
    trailing_text_hidden[gen_step] freshly, so writes are visible
    immediately at subsequent steps.

2.  EOS emission is gated on a `stream_complete` flag (a 1-element
    list shared with the caller). While stream_complete is False, the
    sampler suppresses the codec EOS token so the model can't decide
    speech is over before the caller signals end-of-text. Once
    stream_complete flips True, EOS is allowed and the model finishes
    the trailing portion naturally.

A per-step `on_step` callback gives the caller a chance to mutate the
buffer / flip stream_complete based on its own queue of incoming text.

Compatible with the predictor and talker CUDA graphs from upstream:
trailing_text_hidden's tensor shape and address never change, only
contents, so graph capture/replay remains valid.
"""
from __future__ import annotations

import time
from typing import Callable, List, Optional, Tuple

import torch

from faster_qwen3_tts.predictor_graph import PredictorGraph
from faster_qwen3_tts.sampling import apply_repetition_penalty, sample_logits
from faster_qwen3_tts.talker_graph import TalkerGraph


@torch.inference_mode()
def streaming_fast_generate(
    talker,
    talker_input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    trailing_text_hidden: torch.Tensor,        # PRE-ALLOCATED at max length, mutable
    text_fill_pointer: List[int],              # 1-element mutable: how much real text is in trailing
    stream_complete: List[bool],               # 1-element mutable: True when caller signals end-of-text
    tts_pad_embed: torch.Tensor,
    config,
    predictor_graph: PredictorGraph,
    talker_graph: TalkerGraph,
    on_step: Optional[Callable[[int, int, torch.Tensor, List[int], List[torch.Tensor]], None]] = None,
    on_codec_chunk: Optional[Callable[[torch.Tensor], None]] = None,
    codec_chunk_size: int = 12,
    max_new_tokens: int = 4096,
    min_new_tokens: int = 2,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    do_sample: bool = True,
    repetition_penalty: float = 1.05,
) -> Tuple[Optional[torch.Tensor], dict]:
    """Returns (codec_ids tensor of shape [steps, num_codebooks] or None, timing dict)."""
    eos_id          = config.codec_eos_token_id
    num_code_groups = config.num_code_groups
    vocab_size      = config.vocab_size
    device          = talker_input_embeds.device

    suppress_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    suppress_start = max(0, vocab_size - 1024)
    for i in range(suppress_start, vocab_size):
        if i != eos_id:
            suppress_mask[i] = True

    predictor              = talker.code_predictor
    talker_codec_embed     = talker.get_input_embeddings()
    talker_codec_head      = talker.codec_head
    predictor_codec_embeds = predictor.get_input_embeddings()

    # === PREFILL ===
    t_start = time.time()
    out = talker.forward(
        inputs_embeds=talker_input_embeds,
        attention_mask=attention_mask,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
        trailing_text_hidden=trailing_text_hidden,
        tts_pad_embed=tts_pad_embed,
        generation_step=None,
        past_hidden=None,
        past_key_values=None,
    )
    talker_past_kv = out.past_key_values
    past_hidden    = out.past_hidden
    gen_step       = out.generation_step

    logits = out.logits[:, -1, :]
    suppress_eos = (min_new_tokens > 0) or (not stream_complete[0])
    token = sample_logits(
        logits, temperature=temperature, top_k=top_k, top_p=top_p,
        do_sample=do_sample, suppress_mask=suppress_mask,
        suppress_tokens=[eos_id] if suppress_eos else None,
    )

    prefill_len = talker_graph.prefill_kv(talker_past_kv)
    rope_deltas = getattr(talker, "rope_deltas", None)
    talker_graph.set_generation_state(attention_mask, rope_deltas)
    torch.cuda.synchronize()
    t_prefill = time.time() - t_start

    # === DECODE LOOP ===
    t_decode_start = time.time()
    all_codec_ids = []

    for step_idx in range(max_new_tokens):
        # Hook: caller may mutate trailing_text_hidden / flip stream_complete,
        # and inspect codec produced so far to flush a partial audio chunk.
        if on_step is not None:
            on_step(step_idx, gen_step, trailing_text_hidden, text_fill_pointer, all_codec_ids)

        suppress_eos = (len(all_codec_ids) < min_new_tokens) or (not stream_complete[0])

        if token.item() == eos_id and not suppress_eos:
            break

        last_id_hidden = talker_codec_embed(token.unsqueeze(1))
        pred_input = torch.cat((past_hidden, last_id_hidden), dim=1)
        codebook_token_ids = predictor_graph.run(pred_input)

        all_cb = torch.cat([token.view(1), codebook_token_ids])
        all_codec_ids.append(all_cb.detach())

        codec_hiddens = [last_id_hidden]
        for i in range(num_code_groups - 1):
            codec_hiddens.append(predictor_codec_embeds[i](codebook_token_ids[i].unsqueeze(0).unsqueeze(0)))
        inputs_embeds = torch.cat(codec_hiddens, dim=1).sum(1, keepdim=True)

        if gen_step < trailing_text_hidden.shape[1]:
            inputs_embeds = inputs_embeds + trailing_text_hidden[:, gen_step].unsqueeze(1)
        else:
            inputs_embeds = inputs_embeds + tts_pad_embed

        current_pos = prefill_len + step_idx
        if current_pos >= talker_graph.max_seq_len - 1:
            break

        hidden_states = talker_graph.run(inputs_embeds, position=current_pos)
        logits = talker_codec_head(hidden_states[:, -1, :]).unsqueeze(0)

        if repetition_penalty != 1.0 and len(all_codec_ids) > 0:
            history = torch.stack([c[0] for c in all_codec_ids])
            logits = apply_repetition_penalty(logits, history, repetition_penalty)

        suppress_eos = (len(all_codec_ids) < min_new_tokens) or (not stream_complete[0])
        token = sample_logits(
            logits.squeeze(0), temperature=temperature, top_k=top_k, top_p=top_p,
            do_sample=do_sample, suppress_mask=suppress_mask,
            suppress_tokens=[eos_id] if suppress_eos else None,
        )
        past_hidden = hidden_states[:, -1:, :].clone()
        gen_step += 1

        # Periodic codec-chunk callback so the session can vocode +
        # emit audio while decode is still running. Caller does the
        # vocoder pass on the cumulative codec and tracks what it has
        # already emitted; we just deliver the snapshot.
        if on_codec_chunk is not None \
                and codec_chunk_size > 0 \
                and len(all_codec_ids) > 0 \
                and len(all_codec_ids) % codec_chunk_size == 0:
            on_codec_chunk(torch.stack(all_codec_ids))

    torch.cuda.synchronize()
    t_decode = time.time() - t_decode_start
    n_steps  = len(all_codec_ids)
    timing   = {
        "prefill_ms":  t_prefill * 1000,
        "decode_s":    t_decode,
        "steps":       n_steps,
        "ms_per_step": (t_decode / n_steps * 1000) if n_steps > 0 else 0,
    }
    return (torch.stack(all_codec_ids) if all_codec_ids else None, timing)


def text_to_trailing_embed(model_wrapper, text: str) -> torch.Tensor:
    """
    Tokenize `text` and return its trailing-buffer-style embedding:
        text_projection(text_embeddings(token_ids))     [1, N, hidden_size]

    Skips the chat-template wrappers (the model's own builder strips them
    via input_id[:, 4:-5] before projecting; we just don't add them in the
    first place). Used by the streaming session to compute embeddings for
    user-supplied text chunks at runtime.

    Object hierarchy:
        FasterQwen3TTS  ->  .model is qwen_tts.Qwen3TTSModel
        Qwen3TTSModel   ->  .model is Qwen3TTSForConditionalGeneration (with .talker)
                            .processor.tokenizer is the HF tokenizer
    """
    inference_wrapper = model_wrapper.model
    actual_model      = inference_wrapper.model
    talker            = actual_model.talker
    tokenizer         = inference_wrapper.processor.tokenizer
    device            = next(talker.text_projection.parameters()).device
    ids = torch.tensor(
        [tokenizer.encode(text, add_special_tokens=False)],
        device=device, dtype=torch.long,
    )
    return talker.text_projection(talker.get_text_embeddings()(ids))
