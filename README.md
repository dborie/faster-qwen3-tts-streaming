# faster-qwen3-tts-streaming

> **⚠️ Proof of concept.** Not production-ready. Failure modes are
> minimally handled and the streaming path has known artefacts
> (see the demo recordings below). Use to evaluate the approach,
> not to ship a product.

Streaming text input on top of
[`faster-qwen3-tts`](https://pypi.org/project/faster-qwen3-tts/) (which
itself wraps Alibaba's
[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)): feed text in chunks
while the model is decoding; audio comes out as it goes.

## How it works

`fast_generate` reads from a tensor `trailing_text_hidden[gen_step]`
fresh on every decode step. That tensor is mutable. This package forks
`fast_generate` to:

- write future positions of `trailing_text_hidden` from another thread
  while decode is running;
- suppress EOS until the caller signals end-of-stream, so the model
  doesn't quit when it runs out of supplied text;
- pause the decode loop when text runs out, so the model doesn't drift
  into pad-fed garbage;
- vocode incrementally so audio bytes stream out as codec frames are
  produced.

That's the whole trick.

Supported modes: **custom**, **design**. *Not* clone: ICL clone bakes
text into the prefix at prepare time, no trailing buffer to mutate.
The `demo/` shows what happens if you try anyway.

## Install

```
pip install faster-qwen3-tts-streaming
```

On Windows you'll also need a CUDA build of PyTorch. The default PyPI
wheel is CPU-only and will fail at `device="cuda"`. Install torch from
its own index first:

```
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install faster-qwen3-tts-streaming
```

(Substitute the CUDA tag for your GPU; cu128 covers Blackwell / 50-series.)

## Use

```python
import torch
from faster_qwen3_tts import FasterQwen3TTS
from faster_qwen3_tts_streaming import StreamingSession, SessionConfig

model = FasterQwen3TTS.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device="cuda", dtype=torch.bfloat16,
)

session = StreamingSession(model, SessionConfig(mode="custom", speaker="aiden"))
session.start()

session.feed_text("Hello, ")
# ... do other work; the worker is already generating audio for "Hello, " ...
session.feed_text("how are you today?")
session.complete()

with open("out.pcm", "wb") as f:
    for chunk in session.audio_chunks():
        f.write(chunk)        # 16-bit LE mono PCM at model.sample_rate
session.close()
```

## Demo

`demo/` has a small FastAPI webapp that exercises every Qwen3-TTS
variant + every engine mode side by side. `cd demo && python server.py`,
open http://localhost:8000.

Video walk-through: 

https://github.com/user-attachments/assets/28ff662c-9b76-4211-afb0-0c355b40a41c

### Example outputs

All five files render the same script through the demo on the same
Qwen3-TTS-12Hz-1.7B-CustomVoice checkpoint:

> Wait... did you hear that? Something is moving behind the old wooden
> door. I know we should leave, but what if this is exactly what we
> came here to find? Take a deep breath, stay close, and don't make a
> sound! There it is again... a whisper, or maybe just the wind? No,
> listen carefully. Someone is on the other side, counting down from
> ten. Why would they do that? Nine... eight... seven... Okay, this
> is getting weird! Do we open the door, or do we run? I'm serious.
> Choose now, because whatever is behind that door is not waiting
> much longer! Ready?

All five files use the same prompt; stats are wall-time numbers
from the demo. Click any player to listen inline.

**normal** — 151830 ms gen, 41.92 s audio

https://github.com/user-attachments/assets/1814d739-2336-4220-8113-7e1b8ed65a2c

**faster** — 13372 ms gen, 41.36 s audio

https://github.com/user-attachments/assets/10e528d7-73aa-46f2-8035-37ecdddbb7f2

**faster + streaming** — 14437 ms gen, first audio at 608 ms, 41.36 s audio

https://github.com/user-attachments/assets/8cca24fc-3b22-4716-a3e7-57bce6a1300e

**streaming input** (word-by-word feed) — 52133 ms gen, first audio at 463 ms, 76.56 s audio

https://github.com/user-attachments/assets/9828b32d-1781-423b-a0ae-46ffe58a04a8

**smart input** — 35724 ms gen, first audio at 1664 ms, 42.40 s audio (1 underrun, 7481 ms)

https://github.com/user-attachments/assets/6d53e21c-3b4b-4eae-af76-860372304ef7
