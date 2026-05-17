"""
TimelineRecorder: structured per-event log of a SmartSession run.

Used by tests (and optionally the demo server) to capture a
vertical timeline of every push, flush, audio chunk, and a periodic
tick state snapshot, with playback position computed against the
listener's actual queue. Output is a CSV that can be consumed by
any spreadsheet tool, plus a human-readable text rendering.

Wire-up:

  rec = TimelineRecorder(sample_rate=24000)
  rec.start(state_fn=lambda: {
      "batch_words":      len("".join(session._batch).split()),
      "audio_generated_s": session._total_audio_seconds,
      "samples_queued":   player.samples_queued,
      "samples_consumed": player.samples_consumed,
  })
  # ... drive the session ...
  session.on_flush = lambda text: rec.record_flush(
      text, player.samples_queued)
  for chunk in session.audio_chunks():
      rec.record_chunk(len(chunk))
      player.feed(chunk)
  # ... pusher loop calls rec.record_push(...) per WS frame ...
  rec.stop()
  rec.write_csv(Path("timeline.csv"))
  rec.write_text(Path("timeline.txt"))
"""
import csv
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional


@dataclass
class TimelineRow:
    t_ms              : int
    kind              : str       # "tick" | "push" | "flush" | "chunk" |
                                  # "gen_start" | "gen_end" | "playback_start" |
                                  # "playback_end" | "note"
    detail            : str
    batch_words       : int
    audio_generated_s : float
    listener_played_s : float
    listener_queue_s  : float
    playing_flush     : int       # 1-based flush index the listener is in,
                                  # 0 if not started, -1 if past last flush


class TimelineRecorder:
    """Thread-safe collector. Spawns a tick thread that snapshots state
    every `tick_interval_s` until stop()."""

    def __init__(self, sample_rate: int, tick_interval_s: float = 0.1):
        self.sample_rate         = sample_rate
        self.tick_interval_s     = tick_interval_s
        self._rows               : List[TimelineRow]                 = []
        self._lock               = threading.Lock()
        self._t0                 = time.monotonic()
        self._stop               = threading.Event()
        self._thread             : Optional[threading.Thread]        = None
        self._state_fn           : Optional[Callable[[], dict]]      = None
        # cumulative-sample boundary at each flush; used to map
        # listener position -> "currently playing flush #N".
        self._flush_boundaries   : List[int]                         = []

    # ------- lifecycle -------

    def start(self, state_fn: Callable[[], dict]) -> None:
        self._state_fn = state_fn
        self._thread = threading.Thread(
            target=self._tick_loop, daemon=True, name="timeline-tick")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ------- event recording -------

    def record_push(self, text: str) -> None:
        self._append("push", repr(text))

    def record_flush(self, text: str, samples_queued: int) -> None:
        self._flush_boundaries.append(samples_queued)
        self._append("flush", repr(text))

    def record_chunk(self, n_bytes: int) -> None:
        duration_s = n_bytes / 2 / self.sample_rate
        self._append("chunk", f"{n_bytes}B / {duration_s:.3f}s")

    def record_gen_start(self) -> None:
        self._append("gen_start", "")

    def record_gen_end(self) -> None:
        self._append("gen_end", "")

    def record_playback_start(self) -> None:
        self._append("playback_start", "")

    def record_playback_end(self) -> None:
        self._append("playback_end", "")

    def record_note(self, text: str) -> None:
        self._append("note", text)

    # ------- internals -------

    def _append(self, kind: str, detail: str) -> None:
        t_ms = int((time.monotonic() - self._t0) * 1000)
        st = self._state_fn() if self._state_fn else {}
        samples_queued   = int(st.get("samples_queued",   0))
        samples_consumed = int(st.get("samples_consumed", 0))
        played_s         = samples_consumed / self.sample_rate
        queue_s          = max(0, samples_queued - samples_consumed) / self.sample_rate
        playing_flush    = self._current_flush(samples_consumed)
        with self._lock:
            self._rows.append(TimelineRow(
                t_ms              = t_ms,
                kind              = kind,
                detail            = detail,
                batch_words       = int(st.get("batch_words",       0)),
                audio_generated_s = float(st.get("audio_generated_s", 0.0)),
                listener_played_s = played_s,
                listener_queue_s  = queue_s,
                playing_flush     = playing_flush,
            ))

    def _tick_loop(self) -> None:
        while not self._stop.is_set():
            self._append("tick", "")
            time.sleep(self.tick_interval_s)

    def _current_flush(self, samples_consumed: int) -> int:
        # Each flush event records samples_queued at that instant. Audio
        # for flush #N spans [boundary[N-1] .. boundary[N]] in the
        # cumulative sample stream (boundary[0] = 0 implicitly).
        # samples_consumed within that range -> playing flush #N.
        if not self._flush_boundaries:
            return 0 if samples_consumed == 0 else 1
        # boundary at index i = samples_queued at moment of flush i+1.
        # So flush #1's audio is [0 .. boundary[0]] - but actually
        # boundary[0] was captured AT flush 1, which means "all audio
        # emitted BEFORE flush 1 is part of flushes < 1". flush #1's
        # audio is what arrives AFTER that boundary, until the next
        # flush.
        if samples_consumed < self._flush_boundaries[0]:
            return 0  # listener still on whatever prefix audio (or none)
        for i in range(len(self._flush_boundaries) - 1):
            if samples_consumed < self._flush_boundaries[i + 1]:
                return i + 1   # 1-based flush index
        return len(self._flush_boundaries)

    # ------- output -------

    def rows(self) -> List[TimelineRow]:
        with self._lock:
            return list(self._rows)

    def write_csv(self, path: Path) -> None:
        rows = sorted(self.rows(), key=lambda r: (r.t_ms, r.kind))
        with path.open("w", encoding="utf-8", newline="") as fp:
            # QUOTE_ALL so detail strings containing commas/quotes
            # survive a round-trip through Excel / pandas / awk.
            w = csv.writer(fp, quoting=csv.QUOTE_ALL)
            w.writerow([
                "t_ms", "kind", "detail", "batch_words",
                "audio_gen_s", "listener_played_s", "listener_queue_s",
                "playing_flush",
            ])
            for r in rows:
                w.writerow([
                    r.t_ms, r.kind, r.detail, r.batch_words,
                    f"{r.audio_generated_s:.3f}",
                    f"{r.listener_played_s:.3f}",
                    f"{r.listener_queue_s:.3f}",
                    r.playing_flush,
                ])

    def write_text(self, path: Path) -> None:
        rows = sorted(self.rows(), key=lambda r: (r.t_ms, r.kind))
        lines = []
        lines.append(
            f"{'time':>8} | {'kind':<13} | {'flush#':>6} | "
            f"{'batch':>5} | {'gen':>6} | {'played':>6} | {'queue':>6} | detail")
        lines.append("-" * 110)
        for r in rows:
            lines.append(
                f"{r.t_ms:>6}ms | {r.kind:<13} | "
                f"{r.playing_flush:>6} | {r.batch_words:>4}w | "
                f"{r.audio_generated_s:5.2f}s | "
                f"{r.listener_played_s:5.2f}s | "
                f"{r.listener_queue_s:5.2f}s | "
                f"{r.detail}"
            )
        path.write_text("\n".join(lines), encoding="utf-8")
