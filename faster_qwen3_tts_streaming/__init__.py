"""
Streaming text-input extension for faster-qwen3-tts.

Files:
    decode_loop.py     forked fast_generate with mid-decode trailing-
                       buffer mutation hooks and EOS suppression until
                       the stream completes. Talks to the model
                       directly.
    session.py         worker thread + audio queue + feed_text /
                       complete on top of decode_loop.
    smart_session.py   thin adaptive-batching wrapper on top of
                       StreamingSession; flushes word-by-word while
                       latency-critical, sentence-by-sentence once
                       enough audio is queued ahead of playback.
"""
from .session import SessionConfig, StreamingSession
from .smart_session import SmartSession, SmartSessionConfig

__all__ = [
    "SessionConfig", "StreamingSession",
    "SmartSession", "SmartSessionConfig",
]
