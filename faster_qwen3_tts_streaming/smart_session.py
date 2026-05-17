"""
SmartSession: simple X/Y/Z word-batching wrapper on StreamingSession.

The algorithm in plain language (per user spec):

  1. Bootstrap: accumulate words. Cut at the FIRST sentence
     boundary at a word position >= X (min_initial). If
     word_count reaches Y (max_initial) without such a boundary,
     force-cut at the end of the Y-th word.
  2. Steady state: HOLD all incoming text. Do nothing.
  3. When the measured audio queue ahead of the listener drops to
     within Z seconds of empty, cut at the LAST sentence boundary
     in the held batch.
  4. Repeat until complete() is called.

The audio queue measurement:

  audio_end_wall is the wall-clock instant at which a listener
  (playing at 1x) will have consumed everything we've emitted so
  far. We MEASURE this directly: first_chunk_wall captures the
  moment the first audio byte arrives, and total_audio_seconds
  accumulates each chunk's PCM duration. Then audio_end_wall =
  first_chunk_wall + total_audio_seconds. No prediction, no
  per-word constant - whatever the inner session produces is what
  the queue actually holds.

Knobs:

  smart_buffer:                 on/off
  min_initial_words (X):        bootstrap waits for >= X words AND a
                                sentence boundary
  max_initial_words (Y):        bootstrap failsafe; cut anyway at Y
                                words even without a boundary
  safety_margin_seconds (Z):    steady-state trigger; cut when
                                measured audio queue <= Z seconds
                                ahead of listener

on_flush observability callback fires synchronously with each
flushed string for UI plotting.
"""
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional

from .session import SessionConfig, StreamingSession


SENTENCE_BOUNDARIES = ".!?\n"
POLL_INTERVAL_SECONDS = 0.05


@dataclass
class SmartSessionConfig(SessionConfig):
    smart_buffer: bool = False
    min_initial_words: int = 5            # X
    max_initial_words: int = 30           # Y
    safety_margin_seconds: float = 0.3    # Z


class SmartSession:
    """Simple X/Y/Z batcher. smart_buffer=False is a pass-through."""

    def __init__(self, model_wrapper, config: SmartSessionConfig):
        self._inner          = StreamingSession(model_wrapper, config)
        self._smart_buffer   = config.smart_buffer
        self._min_initial    = max(1, config.min_initial_words)
        self._max_initial    = max(self._min_initial, config.max_initial_words)
        self._safety_margin  = config.safety_margin_seconds

        self._batch: List[str]                 = []
        self._total_words_flushed              = 0
        self._audio_end_wall: Optional[float]  = None
        self._first_chunk_wall: Optional[float] = None
        self._total_audio_seconds              = 0.0
        # Gates the steady-state cut: refuse to flush until the
        # queue has demonstrably grown past 2 * safety_margin since
        # the last flush, so we never cut on a queue that simply
        # hasn't been filled yet.
        self._max_queue_since_flush            = 0.0
        self._lock                             = threading.Lock()
        self._timer: Optional[threading.Timer] = None

        self.on_flush: Optional[Callable[[str], None]] = None

    # ----- public API mirrors StreamingSession -----

    @property
    def sample_rate(self):
        return self._inner.sample_rate

    def start(self) -> None:
        self._inner.start()

    def feed_text(self, text: str) -> None:
        if not text:
            return
        if not self._smart_buffer:
            self._inner.feed_text(text)
            cb = self.on_flush
            if cb is not None:
                try:
                    cb(text)
                except Exception:
                    pass
            return
        with self._lock:
            self._batch.append(text)
            self._evaluate_locked()

    def complete(self) -> None:
        with self._lock:
            self._cancel_timer_locked()
            if self._batch:
                text = "".join(self._batch)
                self._batch.clear()
                self._emit_flush_locked(text)
        self._inner.complete()

    def audio_chunks(self, yield_until_done: bool = True,
                     timeout: Optional[float] = None) -> Iterator[bytes]:
        # Wrap inner's iterator so we observe every chunk on its way
        # out and can update audio_end_wall from real measurements.
        for chunk in self._inner.audio_chunks(
                yield_until_done=yield_until_done, timeout=timeout):
            self._record_audio_chunk(chunk)
            yield chunk

    def _record_audio_chunk(self, chunk: bytes) -> None:
        if not chunk:
            return
        duration = len(chunk) / 2 / self.sample_rate
        with self._lock:
            now = time.monotonic()
            if self._first_chunk_wall is None:
                self._first_chunk_wall = now
            self._total_audio_seconds += duration
            self._audio_end_wall = (
                self._first_chunk_wall + self._total_audio_seconds)

    def close(self) -> None:
        with self._lock:
            self._cancel_timer_locked()
        self._inner.close()

    # ----- algorithm internals -----

    def _evaluate_locked(self) -> None:
        if not self._batch:
            self._cancel_timer_locked()
            return

        joined     = "".join(self._batch)
        word_count = len(joined.split())

        if self._total_words_flushed == 0:
            cut_pos = self._first_boundary_at_or_after_word(
                joined, self._min_initial)
            if cut_pos >= 0:
                self._cut_at_position_locked(joined, cut_pos)
                return
            if word_count >= self._max_initial:
                cut_pos = self._end_of_word(joined, self._max_initial)
                if cut_pos >= 0:
                    self._cut_at_position_locked(joined, cut_pos)
                return
            self._schedule_poll_locked()
            return

        if self._audio_end_wall is None:
            self._schedule_poll_locked()
            return

        now = time.monotonic()
        audio_remaining = self._audio_end_wall - now
        if audio_remaining > self._max_queue_since_flush:
            self._max_queue_since_flush = audio_remaining
        if self._max_queue_since_flush < 2 * self._safety_margin:
            self._schedule_poll_locked()
            return
        if audio_remaining > self._safety_margin:
            self._schedule_poll_locked()
            return

        last_boundary = -1
        for c in SENTENCE_BOUNDARIES:
            idx = joined.rfind(c)
            if idx > last_boundary:
                last_boundary = idx
        if last_boundary < 0:
            # No delimiter in batch yet - hold rather than cut
            # mid-sentence.
            self._schedule_poll_locked()
            return

        self._cut_at_position_locked(joined, last_boundary)

    @staticmethod
    def _first_boundary_at_or_after_word(text: str, min_word: int) -> int:
        """Return char index of the first sentence-boundary character
        that ends a word at position >= min_word (1-based). Returns
        -1 if no such boundary exists yet."""
        word_index = 0
        in_word    = False
        for i, c in enumerate(text):
            if c.isspace():
                in_word = False
                continue
            if not in_word:
                in_word = True
                word_index += 1
            if c in SENTENCE_BOUNDARIES and word_index >= min_word:
                return i
        return -1

    @staticmethod
    def _end_of_word(text: str, word_n: int) -> int:
        """Return char index of the last non-space character of the
        word_n-th word (1-based). Returns -1 if fewer than word_n
        words exist."""
        word_index = 0
        in_word    = False
        last_char  = -1
        for i, c in enumerate(text):
            if c.isspace():
                in_word = False
                continue
            if not in_word:
                in_word = True
                word_index += 1
            if word_index == word_n:
                last_char = i
            elif word_index > word_n:
                break
        return last_char

    def _cut_at_position_locked(self, joined: str, pos: int) -> None:
        text      = joined[:pos + 1]
        remainder = joined[pos + 1:]
        self._batch.clear()
        self._emit_flush_locked(text)
        if remainder.strip():
            self._batch.append(remainder)
            self._schedule_poll_locked()

    def _emit_flush_locked(self, text: str) -> None:
        if not text:
            return
        self._inner.feed_text(text)
        self._total_words_flushed += len(text.split())
        self._max_queue_since_flush = 0.0
        self._cancel_timer_locked()
        cb = self.on_flush
        if cb is not None:
            try:
                cb(text)
            except Exception:
                pass

    def _schedule_poll_locked(self) -> None:
        if self._timer is not None:
            return
        my_timer = threading.Timer(
            POLL_INTERVAL_SECONDS, lambda: self._on_poll(my_timer))
        my_timer.daemon = True
        self._timer = my_timer
        my_timer.start()

    def _on_poll(self, my_timer: threading.Timer) -> None:
        with self._lock:
            if self._timer is not my_timer:
                return
            self._timer = None
            self._evaluate_locked()

    def _cancel_timer_locked(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
