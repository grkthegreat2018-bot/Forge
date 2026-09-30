"""Live status writer for self-play/RSI training — real-time GUI telemetry.

The old status path wrote status.json + heartbeat.json once per EPOCH (4+ min),
so the GUI looked frozen and stalls were invisible. This writer provides:

  - heartbeat.json rewritten every `heartbeat_interval` s from a daemon thread
    (hang detection independent of training-loop liveness)
  - status.json rewritten on phase changes / task completions (throttled),
    carrying live intra-epoch progress: phase, tasks_done/total, throughput,
    ETA, current task, tokens/sec
  - events.jsonl append-only event stream (line-buffered) for the GUI's live
    event feed: phase / curriculum / task_start / round / task_done /
    epoch_done / alert / done

All public methods are thread-safe and never raise into the training loop.

Consumed by forge_gui (EventsReader/StatusReader → SelfPlayPage) which
expects the files under research/checkpoints/self_play/.
"""
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path

from research.json_compat import dumps

logger = logging.getLogger(__name__)


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


class LiveStatusWriter:
    """Writes heartbeat/status/events for live monitoring of self-play."""

    def __init__(self, status_path: str,
                 heartbeat_interval: float = 2.0,
                 min_status_interval: float = 1.0):
        self.status_path = Path(status_path)
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.heartbeat_path = self.status_path.with_name("heartbeat.json")
        self.events_path = self.status_path.with_name("events.jsonl")
        self._hb_interval = max(0.5, heartbeat_interval)
        self._min_status = max(0.2, min_status_interval)

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._status: dict = {"status": "running", "phase": "startup"}
        self._last_status_write = 0.0
        self._start_ts = time.time()
        self._last_progress_ts = time.time()  # for stall detection
        # Long opaque phases (in-process SFT/eval) can't call progress
        # methods — hold the stall flag so the heartbeat doesn't false-positive.
        self._stall_hold = False

        # Throughput tracking (rolling windows)
        self._task_done_ts: deque = deque(maxlen=200)
        self._gen_window: deque = deque(maxlen=200)  # (ts, tokens, gen_ms)
        self._tasks_done = 0
        self._tasks_total = 0
        self._epoch_successes = 0

        # Line-buffered append so the GUI tail sees events immediately.
        self._events_file = open(self.events_path, "a", encoding="utf-8", buffering=1)
        self._hb_thread = threading.Thread(target=self._hb_loop, daemon=True,
                                           name="live-status-heartbeat")
        self._hb_thread.start()

    # ── heartbeat thread ─────────────────────────────────────────────
    # Progress-coupled heartbeat: the thread tracks the last progress
    # timestamp (updated by every public API call) and marks heartbeat.json
    # "stalled" when the training loop hasn't advanced within the stall
    # threshold — a hung loop can't masquerade as alive just because the
    # daemon thread is still running.
    _STALL_THRESHOLD_S = 120.0  # mark stalled if no progress for 2 min

    def _hb_loop(self) -> None:
        while not self._stop.wait(self._hb_interval):
            self._write_heartbeat()

    def _write_heartbeat(self) -> None:
        try:
            now = time.time()
            with self._lock:
                last_progress = self._last_progress_ts
                hold = self._stall_hold
                if hold:
                    # Long phase hold: keep progress fresh so stall
                    # detection doesn't fire during in-process SFT/eval.
                    self._last_progress_ts = now
                    last_progress = now
            stall_age = now - last_progress
            hb = {"ts": now}
            if stall_age > self._STALL_THRESHOLD_S:
                hb["stalled"] = True
                hb["stall_age_s"] = round(stall_age, 1)
            _atomic_write(self.heartbeat_path, dumps(hb))
        except Exception as e:
            logger.debug("heartbeat write failed: %s", e)

    def stop_requested(self) -> bool:
        """Check for the cooperative STOP_REQUESTED sentinel file.

        The GUI writes this sentinel next to status.json when the user
        clicks Stop. The training loop polls this method at epoch/task
        boundaries to shut down gracefully — reliable on Windows where
        SIGTERM delivery from taskkill is unreliable.
        """
        sentinel = self.status_path.parent / "STOP_REQUESTED"
        try:
            return sentinel.is_file()
        except Exception:
            return False

    # ── status.json ──────────────────────────────────────────────────
    def _write_status(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_status_write) < self._min_status:
            return
        self._last_status_write = now
        try:
            _atomic_write(self.status_path, dumps(self._status, indent=2))
        except Exception as e:
            logger.debug("status write failed: %s", e)

    def _refresh_rates(self) -> None:
        """Recompute throughput/ETA fields from rolling windows."""
        now = time.time()
        # tasks/min over last 120s
        recent = [t for t in self._task_done_ts if now - t <= 120.0]
        if len(recent) >= 2:
            span = max(now - recent[0], 1.0)
            self._status["tasks_per_min"] = round(len(recent) / span * 60.0, 2)
        else:
            self._status["tasks_per_min"] = 0.0
        # gen tok/s over rolling window
        if self._gen_window:
            cutoff = now - 120.0
            toks = sum(t for ts, t, _ in self._gen_window if ts >= cutoff)
            ms = sum(m for ts, _, m in self._gen_window if ts >= cutoff)
            self._status["gen_tok_s"] = round(toks / (ms / 1000.0), 1) if ms > 0 else 0.0
        # ETA from tasks remaining / current rate
        rate = self._status.get("tasks_per_min", 0.0)
        remaining = max(self._tasks_total - self._tasks_done, 0)
        self._status["eta_s"] = round(remaining / (rate / 60.0), 1) if rate > 0 else None
        self._status["tasks_done"] = self._tasks_done
        self._status["tasks_total"] = self._tasks_total
        self._status["epoch_successes"] = self._epoch_successes
        self._status["elapsed_s"] = round(now - self._start_ts, 1)
        self._status["ts"] = now

    # ── public API ───────────────────────────────────────────────────
    def _touch_progress(self) -> None:
        """Mark that the training loop made progress (for stall detection)."""
        self._last_progress_ts = time.time()

    def set_phase(self, phase: str, detail: str = "",
                  stall_hold: bool = False, **fields) -> None:
        """Transition the run phase (proposing/solving/finetune/evaluate/...).

        ``stall_hold=True`` suspends stall detection for phases that make
        no progress calls (in-process SFT, eval) — the heartbeat stays fresh
        but the GUI still sees the phase change.
        """
        with self._lock:
            self._touch_progress()
            self._stall_hold = stall_hold
            self._status["phase"] = phase
            self._status["phase_detail"] = detail
            self._status.update(fields)
            self._emit_locked("phase", phase=phase, detail=detail)
            self._refresh_rates()
            self._write_status(force=True)

    def update(self, **fields) -> None:
        """Merge fields into status.json (throttled writes)."""
        with self._lock:
            self._touch_progress()
            self._status.update(fields)
            self._refresh_rates()
            self._write_status()

    def event(self, kind: str, **fields) -> None:
        """Emit a raw event to events.jsonl."""
        with self._lock:
            self._touch_progress()
            self._emit_locked(kind, **fields)

    def task_started(self, task: str, idx: int, total: int, **fields) -> None:
        with self._lock:
            self._touch_progress()
            self._tasks_total = total
            self._status["current_task"] = task[:80]
            self._emit_locked("task_start", task=task[:120], idx=idx, total=total, **fields)
            self._refresh_rates()
            self._write_status()

    def round_done(self, task_idx: int, round_num: int, success: bool,
                   quality: float = 0.0, gen_ms: float = 0.0, exec_ms: float = 0.0,
                   tokens: int = 0, error: str = "", round_active: int = 0) -> None:
        with self._lock:
            self._touch_progress()
            if tokens or gen_ms:
                self._gen_window.append((time.time(), tokens, gen_ms))
            self._status["round_active"] = round_active
            self._emit_locked("round", task_idx=task_idx, round=round_num,
                              success=success, quality=round(quality, 4),
                              gen_ms=round(gen_ms, 1), exec_ms=round(exec_ms, 1),
                              error=error[:200])
            self._refresh_rates()
            self._write_status()

    def task_done(self, task_idx: int, task: str, success: bool,
                  rounds_used: int = 0, best_quality: float = 0.0,
                  preview: str = "", expected: str = "") -> None:
        with self._lock:
            self._touch_progress()
            self._tasks_done += 1
            if success:
                self._epoch_successes += 1
            self._task_done_ts.append(time.time())
            # preview/expected let spectators see what the model actually
            # produced vs the verified answer — the core spectation signal
            self._emit_locked("task_done", task_idx=task_idx, task=task[:120],
                              success=success, rounds_used=max(int(rounds_used), 0),
                              best_quality=round(best_quality, 4),
                              preview=preview[:200], expected=expected[:80])
            self._refresh_rates()
            self._write_status(force=True)

    def curriculum_progress(self, proposed: int, validated: int, parse_failed: int) -> None:
        with self._lock:
            self._touch_progress()
            self._emit_locked("curriculum", proposed=proposed, validated=validated,
                              parse_failed=parse_failed)
            self._status["curriculum_proposed"] = proposed
            self._status["curriculum_validated"] = validated
            self._write_status()

    def epoch_done(self, epoch: int, n_epochs: int, **metrics) -> None:
        with self._lock:
            self._touch_progress()
            self._emit_locked("epoch_done", epoch=epoch, n_epochs=n_epochs, **metrics)
            self._tasks_done = 0
            self._epoch_successes = 0
            self._refresh_rates()
            self._write_status(force=True)

    def alert(self, level: str, msg: str) -> None:
        with self._lock:
            self._touch_progress()
            self._emit_locked("alert", level=level, msg=msg[:300])

    def _emit_locked(self, kind: str, **fields) -> None:
        try:
            rec = {"ts": round(time.time(), 3), "kind": kind}
            rec.update(fields)
            self._events_file.write(dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug("event write failed: %s", e)

    def close(self, status: str = "done", reason: str = "") -> None:
        """Final write + stop the heartbeat thread."""
        with self._lock:
            self._status["status"] = status
            self._status["phase"] = status
            self._emit_locked("done", reason=reason[:300])
            self._write_status(force=True)
            self._write_heartbeat()
        self._stop.set()
        self._hb_thread.join(timeout=2.0)
        try:
            self._events_file.close()
        except Exception:
            pass
