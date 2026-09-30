"""Crash recovery and persistence for ForgeEngine.

Provides:
  - CrashRecoveryManager: signal handlers (SIGTERM/SIGINT) + atexit cleanup
  - Generation checkpointing: periodically saves partial generation output
    to disk during long generations, so a crash mid-generation loses at
    most ``checkpoint_interval`` tokens of work.
  - KV cache snapshotting: saves KV cache state to compressed disk files
    so long conversations can be resumed after crash/restart.
  - Event log persistence: flushes EventLog to disk on crash, so
    diagnostics survive process death.
  - Output history persistence: flushes OutputHistory to disk on crash.

All disk I/O uses zstd compression (via the ``zstandard`` package, with
a fallback to uncompressed if unavailable) to maximize drive throughput
and minimize disk space usage.

Usage (automatic via ForgeEngine):
    engine = ForgeEngine.from_checkpoint(...)
    # CrashRecoveryManager is installed automatically in __init__
    # On SIGTERM/SIGINT: flushes logs, saves KV snapshot, saves partial output
    # On atexit: same cleanup
    # On next start: engine.recover() restores last state
"""
from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from forge.engine.forge_engine import ForgeEngine

_DEFAULT_RECOVERY_DIR = ".forge_recovery"
_CHECKPOINT_INTERVAL = 32  # save partial output every N tokens
_KV_SNAPSHOT_INTERVAL = 256  # save KV snapshot every N tokens
_MAX_SNAPSHOTS = 3  # keep last N KV snapshots, rotate


def _try_import_zstd():
    """Try to import zstandard for compressed I/O."""
    try:
        import zstandard as zstd
        return zstd
    except ImportError:
        return None


_zstd = _try_import_zstd()


def _compress(data: bytes) -> bytes:
    """Compress bytes with zstd (fallback: uncompressed)."""
    if _zstd is not None:
        return _zstd.compress(data, level=3)
    return data


def _decompress(data: bytes) -> bytes:
    """Decompress bytes (auto-detects zstd vs raw)."""
    if _zstd is not None and data[:4] == b"\x28\xb5\x2f\xfd":
        return _zstd.decompress(data)
    return data


class CrashRecoveryManager:
    """Manages crash recovery for a ForgeEngine instance.

    Installs signal handlers and atexit cleanup. Provides methods for
    checkpointing generation state and snapshotting KV cache to disk.

    All files are stored in ``recovery_dir`` (default: ``.forge_recovery/``)
    using zstd compression for minimal disk I/O.
    """

    def __init__(self, engine: ForgeEngine,
                 recovery_dir: str = _DEFAULT_RECOVERY_DIR,
                 enabled: bool = True):
        self.engine = engine
        self.recovery_dir = Path(recovery_dir)
        self.enabled = enabled
        self._installed = False
        self._token_count = 0
        self._last_partial_output: str | None = None
        self._last_prompt: str | None = None
        self._snapshot_counter = 0
        self._lock = threading.Lock()

        if enabled:
            self._install()

    def _install(self):
        """Install signal handlers and atexit cleanup."""
        if self._installed:
            return
        self.recovery_dir.mkdir(parents=True, exist_ok=True)

        # Weakref callbacks: atexit and the signal table are GC roots —
        # registering bound methods directly would pin the manager (and,
        # through ``self.engine``, the entire engine + model) alive until
        # process exit, defeating ForgeEngine.__del__'s VRAM cleanup.
        import atexit
        import weakref
        mgr_ref = weakref.ref(self)

        def _atexit_cleanup():
            mgr = mgr_ref()
            if mgr is not None:
                mgr._cleanup()
        atexit.register(_atexit_cleanup)
        self._atexit_fn = _atexit_cleanup  # for explicit unregister

        # Signal handlers for graceful shutdown
        old_sigterm = signal.getsignal(signal.SIGTERM)
        old_sigint = signal.getsignal(signal.SIGINT)

        def _handler(signum, frame):
            mgr = mgr_ref()
            if mgr is None:
                return
            mgr._log(f"Signal {signum} received, flushing recovery state...")
            mgr._cleanup()
            # Re-raise to default handler so the process actually exits
            if signum == signal.SIGINT:
                signal.signal(signal.SIGINT, old_sigint)
                raise KeyboardInterrupt
            elif signum == signal.SIGTERM:
                signal.signal(signal.SIGTERM, old_sigterm)
                # Default SIGTERM terminates the process
                os._exit(128 + signum)

        try:
            signal.signal(signal.SIGTERM, _handler)
        except (ValueError, OSError):
            pass  # not in main thread
        try:
            signal.signal(signal.SIGINT, _handler)
        except (ValueError, OSError):
            pass  # not in main thread

        self._installed = True
        self._log("Crash recovery enabled", recovery_dir=str(self.recovery_dir))

    def close(self):
        """Disable recovery and unregister the atexit hook.

        Explicit teardown for engine.close()-style lifecycles — signal
        handlers are left installed (harmless no-ops once the manager is
        dead) since restoring them safely needs the main thread.
        """
        self.enabled = False
        if self._installed:
            import atexit
            try:
                atexit.unregister(self._atexit_fn)
            except Exception:
                pass
            self._installed = False

    def _log(self, message: str, **data):
        """Log via the engine's event log if available."""
        if hasattr(self.engine, '_log'):
            self.engine._log(message, source="recovery", **data)

    def checkpoint_generation(self, prompt: str, partial_output: str,
                              token_count: int):
        """Save partial generation output to disk.

        Called periodically during generation. On crash, the last
        checkpointed output can be recovered via ``recover()``.
        """
        if not self.enabled:
            return
        with self._lock:
            self._last_prompt = prompt
            self._last_partial_output = partial_output
            self._token_count = token_count

            if token_count % _CHECKPOINT_INTERVAL != 0:
                return

            data = {
                "prompt": prompt,
                "partial_output": partial_output,
                "token_count": token_count,
                "timestamp": time.time(),
            }
            blob = _compress(json.dumps(data, ensure_ascii=False).encode())
            path = self.recovery_dir / "generation_checkpoint.zst"
            try:
                path.write_bytes(blob)
            except OSError as e:
                self._log(f"Failed to save generation checkpoint: {e}",
                          level="warn")

    def snapshot_kv_cache(self, token_count: int):
        """Snapshot KV cache state to compressed disk file.

        Called periodically during generation. On crash, the last
        snapshot can be restored via ``recover()``.
        """
        if not self.enabled or not self.engine.kv_cache:
            return
        if token_count % _KV_SNAPSHOT_INTERVAL != 0:
            return

        with self._lock:
            try:
                kv = self.engine.kv_cache
                # Extract K/V tensors from the cache
                if not hasattr(kv, 'k_cache') or kv.k_cache is None:
                    return

                k_data = kv.k_cache.cpu().contiguous()
                v_data = kv.v_cache.cpu().contiguous()
                # Store as compressed torch save
                import io

                import torch
                buf = io.BytesIO()
                torch.save({
                    "k_cache": k_data,
                    "v_cache": v_data,
                    "seq_len": kv.seq_len,
                    "token_count": token_count,
                }, buf)
                blob = _compress(buf.getvalue())

                # Rotate snapshots
                slot = self._snapshot_counter % _MAX_SNAPSHOTS
                self._snapshot_counter += 1
                path = self.recovery_dir / f"kv_snapshot_{slot}.zst"
                path.write_bytes(blob)
            except Exception as e:
                self._log(f"Failed to snapshot KV cache: {e}",
                          level="warn")

    def _cleanup(self):
        """Flush all in-memory state to disk (called on exit/crash)."""
        if not self.enabled:
            return
        with self._lock:
            # Save final generation state
            if self._last_partial_output is not None:
                data = {
                    "prompt": self._last_prompt,
                    "partial_output": self._last_partial_output,
                    "token_count": self._token_count,
                    "timestamp": time.time(),
                    "final": True,
                }
                blob = _compress(json.dumps(data, ensure_ascii=False).encode())
                try:
                    (self.recovery_dir / "generation_final.zst").write_bytes(blob)
                except OSError:
                    pass

            # Flush event log
            if hasattr(self.engine, 'events'):
                events = self.engine.events.to_list()
                blob = _compress(json.dumps(events).encode())
                try:
                    (self.recovery_dir / "event_log.zst").write_bytes(blob)
                except OSError:
                    pass

            # Flush output history
            if hasattr(self.engine, 'outputs'):
                outputs = self.engine.outputs.read_output(n=0)
                blob = _compress(json.dumps(outputs).encode())
                try:
                    (self.recovery_dir / "output_history.zst").write_bytes(blob)
                except OSError:
                    pass

    def recover(self) -> dict[str, Any] | None:
        """Recover state from disk after a crash.

        Returns a dict with recovered state, or None if no recovery data.
        """
        if not self.enabled:
            return None

        result: dict[str, Any] = {}

        # Recover generation state
        for name in ("generation_final.zst", "generation_checkpoint.zst"):
            path = self.recovery_dir / name
            if path.exists():
                try:
                    data = json.loads(_decompress(path.read_bytes()))
                    result["generation"] = data
                    break
                except Exception:
                    logger.debug("Failed to recover generation state from %s", name, exc_info=True)

        # Recover event log
        path = self.recovery_dir / "event_log.zst"
        if path.exists():
            try:
                result["event_log"] = json.loads(_decompress(path.read_bytes()))
            except Exception:
                logger.debug("Failed to recover event log", exc_info=True)

        # Recover output history
        path = self.recovery_dir / "output_history.zst"
        if path.exists():
            try:
                result["output_history"] = json.loads(_decompress(path.read_bytes()))
            except Exception:
                logger.debug("Failed to recover output history", exc_info=True)

        # Recover KV cache (find most recent snapshot)
        best_snapshot = None
        best_token_count = -1
        for slot in range(_MAX_SNAPSHOTS):
            path = self.recovery_dir / f"kv_snapshot_{slot}.zst"
            if path.exists():
                try:
                    import io

                    import torch
                    blob = _decompress(path.read_bytes())
                    # Try weights_only=True first (safe, blocks arbitrary code),
                    # fall back to weights_only=False for older snapshots
                    try:
                        snapshot = torch.load(io.BytesIO(blob), weights_only=True)
                    except Exception:
                        snapshot = torch.load(io.BytesIO(blob), weights_only=False)
                    if snapshot["token_count"] > best_token_count:
                        best_snapshot = snapshot
                        best_token_count = snapshot["token_count"]
                except Exception:
                    logger.debug("Failed to recover KV snapshot from slot %d", slot, exc_info=True)
        if best_snapshot is not None:
            result["kv_snapshot"] = best_snapshot

        return result if result else None

    def clear(self):
        """Remove all recovery files."""
        if self.recovery_dir.exists():
            for f in self.recovery_dir.glob("*.zst"):
                try:
                    f.unlink()
                except OSError:
                    pass
