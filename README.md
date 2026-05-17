# faster-qwen3-tts-streaming

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
