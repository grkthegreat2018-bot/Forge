"""Incremental events.jsonl tailer for the Self-Play live page.

Tails events.jsonl files under research/checkpoints/** and provides:
  - poll(): new events since last call (oldest-first)
  - all_events(): bounded rolling buffer (deque maxlen ~2000)
  - latest_status(): cheap status.json read with mtime cache

Robust to truncation/rotation: offset reset on shrink, ctime change
(delete+recreate), and torn-read recovery for same-inode O_TRUNC rewrites
that grow the file past the old offset (rewinds to the containing line
rather than dropping the record). Partial last lines are held until a
newline arrives. Designed to be called every ~500ms from the UI thread
without blocking.
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path

from .status_reader import project_root

logger = logging.getLogger(__name__)


def _is_json(raw: bytes) -> bool:
    try:
        json.loads(raw)
        return True
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return False


class _EventSource:
    __slots__ = ("mtime", "offset", "partial", "path", "ctime")

    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.partial = b""
        self.mtime = 0.0
        self.ctime = 0.0


class EventsReader:
    """Tails events.jsonl + reads status.json for the Self-Play page."""

    def __init__(self, search_dirs: list[Path] | None = None,
                 max_events: int = 2000,
                 rescan_interval: float = 5.0) -> None:
        self._search_dirs = search_dirs or [project_root() / "research" / "checkpoints"]
        self._sources: dict[str, _EventSource] = {}
        self._buffer: deque[dict] = deque(maxlen=max_events)
        self._last_rescan = 0.0
        self._rescan_interval = rescan_interval
        # status.json cache
        self._status_path: Path | None = None
        self._status_mtime = 0.0
        self._status_cache: dict | None = None

    def _discover(self) -> None:
        """Find events.jsonl files under search dirs (called at most every rescan_interval).

        On first discovery of a file, back-fills the last ~200 events so the
        GUI shows recent history immediately instead of a blank feed.
        """
        for d in self._search_dirs:
            if not d.is_dir():
                continue
            for p in d.rglob("events.jsonl"):
                key = str(p)
                if key not in self._sources:
                    try:
                        st = p.stat()
                        src = _EventSource(p)
                        # Back-fill: read last ~50KB of the file to populate
                        # recent history (roughly 200-500 events at ~200B each).
                        backfill = min(st.st_size, 50_000)
                        src.offset = st.st_size - backfill
                        src.mtime = st.st_mtime
                        src.ctime = st.st_ctime
                        self._sources[key] = src
                    except OSError:
                        pass

    def poll(self) -> list[dict]:
        """Read new events from all sources. Returns oldest-first list."""
        now = time.time()
        if now - self._last_rescan > self._rescan_interval:
            self._last_rescan = now
            self._discover()

        new_events: list[dict] = []
        for src in list(self._sources.values()):
            try:
                st = src.path.stat()
            except OSError:
                continue
            size = st.st_size
            if size < src.offset or (src.ctime and
                                     st.st_ctime != src.ctime):
                # Truncated, rotated, or delete+recreated — restart from 0
                src.offset = 0
                src.partial = b""
                src.ctime = st.st_ctime
            if size == src.offset:
                continue

            for _attempt in (0, 1):
                read_from = src.offset
                try:
                    with open(src.path, "rb") as f:
                        f.seek(read_from)
                        chunk = f.read(size - read_from)
                    src.offset = size
                except OSError as e:
                    logger.debug("events read failed %s: %s", src.path, e)
                    break

                had_partial = bool(src.partial)
                data = src.partial + chunk
                lines = data.split(b"\n")
                # last element may be partial (no trailing \n)
                src.partial = lines.pop()

                # Torn-read detection: a same-inode rewrite (O_TRUNC then
                # write *larger* than the old offset) leaves offset parked
                # mid-line — the first line is a JSON fragment. If read_from
                # wasn't at a line boundary and the first line won't parse,
                # rewind to the containing line's start and re-read once.
                if (not _attempt and not had_partial and read_from > 0):
                    first = next((l for l in lines if l.strip()), b"")
                    if first and not _is_json(first) and \
                            self._rewind_to_line_start(src, read_from):
                        continue
                for raw in lines:
                    if not raw.strip():
                        continue
                    try:
                        ev = json.loads(raw)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    self._buffer.append(ev)
                    new_events.append(ev)
                break

        return new_events

    @staticmethod
    def _rewind_to_line_start(src: _EventSource, read_from: int) -> bool:
        """Rewind src.offset to the start of the line containing
        ``read_from`` — only when read_from is provably mid-line.

        The byte before a true line boundary is always ``\\n``; if it
        isn't, the previous read started inside a record (same-inode
        O_TRUNC rewrite larger than the old offset). Returns False when
        read_from is already at a boundary (genuinely corrupt line —
        skip it, don't rewind) or no boundary exists within the window.
        """
        try:
            window = min(read_from, 8192)
            with open(src.path, "rb") as f:
                f.seek(read_from - window)
                blob = f.read(window)
        except OSError:
            return False
        if not blob or blob.endswith(b"\n"):
            return False
        idx = blob.rfind(b"\n")
        if idx < 0:
            # no boundary in the window — only safe to rewind to 0 when
            # the window reached the file start (containing line begins
            # at offset 0); a >8KB line mid-file is unrecoverable
            if read_from - window > 0:
                return False
            src.offset = 0
        else:
            src.offset = read_from - window + idx + 1
        src.partial = b""
        return True

    def all_events(self) -> list[dict]:
        """Return the bounded rolling buffer (oldest-first, up to maxlen)."""
        return list(self._buffer)

    def latest_status(self) -> dict | None:
        """Cheap read of the newest status.json under checkpoints/self_play/."""
        # Find the status file (prefer self_play/ subdir)
        candidates: list[Path] = []
        for d in self._search_dirs:
            if not d.is_dir():
                continue
            sp = d / "self_play" / "status.json"
            if sp.is_file():
                candidates.append(sp)
            else:
                for p in d.rglob("status.json"):
                    if "self_play" in p.parent.name.lower():
                        candidates.append(p)

        if not candidates:
            self._status_path = None
            self._status_cache = None
            return None

        # Pick the most recently modified
        best = max(candidates, key=lambda p: p.stat().st_mtime)
        try:
            mtime = best.stat().st_mtime
        except OSError:
            return self._status_cache

        if best == self._status_path and mtime == self._status_mtime and self._status_cache is not None:
            return self._status_cache

        try:
            with open(best, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._status_path = best
                self._status_mtime = mtime
                self._status_cache = data
                return data
        except (json.JSONDecodeError, OSError) as e:
            logger.debug("status read failed %s: %s", best, e)

        return self._status_cache

    def heartbeat_age(self) -> float | None:
        """Seconds since the last heartbeat.json write (None if no heartbeat file)."""
        if self._status_path is None:
            self.latest_status()
        if self._status_path is None:
            return None
        hb = self._status_path.with_name("heartbeat.json")
        if not hb.is_file():
            return None
        try:
            with open(hb, encoding="utf-8") as f:
                data = json.load(f)
            ts = float(data.get("ts", 0))
            return time.time() - ts
        except (json.JSONDecodeError, OSError, ValueError):
            return None

    def heartbeat_stalled(self) -> bool | None:
        """Check if the training loop has stalled (progress-coupled heartbeat).

        Returns True if the heartbeat writer detected no training-loop
        progress within its stall threshold (the heartbeat.json contains
        a "stalled": true flag). Returns False if the heartbeat is fresh
        and progress is being made. Returns None if no heartbeat file
        exists or it can't be read.
        """
        if self._status_path is None:
            self.latest_status()
        if self._status_path is None:
            return None
        hb = self._status_path.with_name("heartbeat.json")
        if not hb.is_file():
            return None
        try:
            with open(hb, encoding="utf-8") as f:
                data = json.load(f)
            return bool(data.get("stalled", False))
        except (json.JSONDecodeError, OSError, ValueError):
            return None
