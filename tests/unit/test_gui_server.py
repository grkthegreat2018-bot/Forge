"""Unit tests for forge_gui_server services â€” hub, engine lease, procs.

CPU-only: no torch, no real checkpoint. Engine internals are stubbed.
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
import time

import pytest

from forge_gui_server.hub import EventHub
from forge_gui_server.services.engine_rt import EngineService
from forge_gui_server.services.procs import ProcessService


# â”€â”€ EventHub â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_hub_publish_replay():
    hub = EventHub()
    hub.publish("engine", {"state": "ready"})
    hub.publish("gpu", {"vram": 100})
    q = hub.subscribe(replay=True)
    e1 = q.get_nowait()
    e2 = q.get_nowait()
    assert e1["type"] == "engine" and e1["seq"] == 1
    assert e2["type"] == "gpu" and e2["seq"] == 2
    assert q.empty()


def test_hub_fanout_and_drops_slow():
    hub = EventHub()
    q = hub.subscribe(replay=False)
    # overflow the subscriber queue â€” it gets dropped, publisher unaffected
    for _ in range(600):
        hub.publish("spam", {})
    assert hub.subscriber_count == 0
    hub.publish("still", {"ok": True})  # must not raise


def test_hub_unsubscribe():
    hub = EventHub()
    q = hub.subscribe(replay=False)
    hub.unsubscribe(q)
    hub.publish("x", {})
    assert q.empty()


# â”€â”€ EngineService lease semantics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class _FakeEngine:
    def sleep(self):
        pass


def _ready_service() -> EngineService:
    svc = EngineService(EventHub())
    svc._engine = _FakeEngine()
    svc._state = "ready"
    svc._info = {"config_name": "test"}
    return svc


def test_lease_requires_ready():
    svc = EngineService(EventHub())
    with pytest.raises(RuntimeError, match="not loaded"):
        with svc.acquire(timeout_s=0.1):
            pass


def test_lease_serializes_generation():
    svc = _ready_service()
    with svc.acquire(timeout_s=1.0) as eng:
        assert isinstance(eng, _FakeEngine)
        # second acquire blocks and times out â€” generation is serialized
        with pytest.raises(RuntimeError, match="busy"):
            with svc.acquire(timeout_s=0.2):
                pass
    # lease released â€” next acquire works again
    with svc.acquire(timeout_s=1.0):
        pass


def test_lease_concurrent_threads():
    svc = _ready_service()
    order: list[str] = []

    def hold(tag, delay):
        with svc.acquire(timeout_s=5.0):
            order.append(f"{tag}-in")
            time.sleep(delay)
            order.append(f"{tag}-out")

    t1 = threading.Thread(target=hold, args=("a", 0.15))
    t2 = threading.Thread(target=hold, args=("b", 0.01))
    t1.start()
    time.sleep(0.03)  # ensure t1 wins the lock
    t2.start()
    t1.join(); t2.join()
    assert order == ["a-in", "a-out", "b-in", "b-out"]


def test_engine_load_unload_cycle(monkeypatch):
    svc = EngineService(EventHub())
    monkeypatch.setattr(
        svc, "_load_blocking",
        lambda *a, **k: (_FakeEngine(), {"config_name": "fake", "load_s": 0.01}))

    async def run():
        loop = asyncio.get_running_loop()
        svc.bind_loop(loop)
        svc._set_state("loading")
        svc._load_seq += 1
        await svc._load_async(svc._load_seq, loop, "ckpt", "fake",
                              None, None, None)
        assert svc.is_ready()
        assert svc.info["config_name"] == "fake"
        assert svc.snapshot()["state"] == "ready"
        svc.unload()
        assert svc.state == "idle"
        assert not svc.is_ready()

    asyncio.run(run())


def test_engine_load_failure_sets_error(monkeypatch):
    svc = EngineService(EventHub())

    def boom(*a, **k):
        raise RuntimeError("checkpoint not found")

    monkeypatch.setattr(svc, "_load_blocking", boom)

    async def run():
        loop = asyncio.get_running_loop()
        svc.bind_loop(loop)
        svc._set_state("loading")
        svc._load_seq += 1
        await svc._load_async(svc._load_seq, loop, "ckpt", "fake",
                              None, None, None)
        assert svc.state == "error"
        assert "checkpoint not found" in svc.error

    asyncio.run(run())


def test_engine_load_queues_different_model(monkeypatch):
    """A different-model request during an in-flight load is queued, not
    dropped; it runs after the first load settles."""
    svc = EngineService(EventHub())
    started = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    def slow_load(ckpt, config_name, *a, **k):
        calls.append(config_name)
        started.set()
        release.wait(timeout=10)
        return _FakeEngine(), {"config_name": config_name,
                               "checkpoint": ckpt}

    monkeypatch.setattr(svc, "_load_blocking", slow_load)

    async def run():
        loop = asyncio.get_running_loop()
        svc.bind_loop(loop)
        svc.load("a.safetensors", "model_a")
        await asyncio.sleep(0.05)
        assert svc.state == "loading"
        assert started.wait(timeout=5)
        # Different model mid-load -> queued, not dropped
        svc.load("b.safetensors", "model_b")
        assert svc._queued_load is not None
        release.set()
        # wait for both loads to settle
        for _ in range(100):
            if svc.is_ready() and svc.info.get("config_name") == "model_b":
                break
            await asyncio.sleep(0.05)
        assert calls == ["model_a", "model_b"]
        assert svc.info["config_name"] == "model_b"

    asyncio.run(run())


def test_engine_load_same_model_inflight_dedup(monkeypatch):
    """Identical request during an in-flight load is deduped."""
    svc = EngineService(EventHub())
    release = threading.Event()
    calls: list[str] = []

    def slow_load(ckpt, config_name, *a, **k):
        calls.append(config_name)
        release.wait(timeout=10)
        return _FakeEngine(), {"config_name": config_name,
                               "checkpoint": ckpt}

    monkeypatch.setattr(svc, "_load_blocking", slow_load)

    async def run():
        loop = asyncio.get_running_loop()
        svc.bind_loop(loop)
        svc.load("a.safetensors", "model_a")
        await asyncio.sleep(0.05)
        svc.load("a.safetensors", "model_a")  # dup — ignored
        svc.load("a.safetensors", "model_a")  # dup — ignored
        release.set()
        for _ in range(100):
            if svc.is_ready():
                break
            await asyncio.sleep(0.05)
        assert calls == ["model_a"]
        assert svc._queued_load is None

    asyncio.run(run())


def test_engine_unload_during_load_discards(monkeypatch):
    """unload() while a load is in flight must not leave a zombie engine."""
    svc = EngineService(EventHub())
    started = threading.Event()
    release = threading.Event()

    def slow_load(*a, **k):
        started.set()
        release.wait(timeout=10)
        return _FakeEngine(), {"config_name": "late"}

    monkeypatch.setattr(svc, "_load_blocking", slow_load)

    async def run():
        loop = asyncio.get_running_loop()
        svc.bind_loop(loop)
        svc.load("a.safetensors", "model_a")
        await asyncio.sleep(0.05)
        assert started.wait(timeout=5)
        svc.unload()
        release.set()
        await asyncio.sleep(0.3)
        assert svc.state == "idle"
        assert svc._engine is None
        assert not svc.is_ready()

    asyncio.run(run())


def test_engine_reload_same_model_noop(monkeypatch):
    """Bare re-request for the resident model is a no-op."""
    svc = _ready_service()
    ckpt = svc._resolve_ckpt("a.safetensors")
    svc._info = {"checkpoint": ckpt, "config_name": "test"}
    calls: list = []
    monkeypatch.setattr(svc, "_load_blocking",
                        lambda *a, **k: calls.append(1) or (_FakeEngine(), {}))
    svc.load("a.safetensors", "test")
    assert calls == []
    assert svc.state == "ready"


def test_reactivate_requires_ready():
    svc = EngineService(EventHub())
    svc.reactivate({"kv_cache": "paged"})
    assert svc.state == "idle"  # unchanged; error published to hub


# â”€â”€ ProcessService â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_process_lifecycle():
    svc = ProcessService(EventHub())

    async def run():
        loop = asyncio.get_running_loop()
        svc.bind_loop(loop)
        tid = svc.launch("echo-test",
                         [sys.executable, "-c", "print('hello');print('bye')"])
        assert tid in svc.tasks
        # wait for the subprocess to finish
        for _ in range(100):
            if svc.tasks[tid].status == "done":
                break
            await asyncio.sleep(0.1)
        info = svc.tasks[tid]
        assert info.status == "done"
        assert info.exit_code == 0
        assert "hello" in info.lines and "bye" in info.lines
        # finished tasks are removable; remove again -> False
        assert svc.remove(tid)
        assert tid not in svc.tasks

    asyncio.run(run())


def test_process_crash_status():
    svc = ProcessService(EventHub())

    async def run():
        loop = asyncio.get_running_loop()
        svc.bind_loop(loop)
        tid = svc.launch("crash-test",
                         [sys.executable, "-c", "import sys;sys.exit(3)"])
        for _ in range(100):
            if svc.tasks[tid].status == "crashed":
                break
            await asyncio.sleep(0.1)
        assert svc.tasks[tid].status == "crashed"
        assert svc.tasks[tid].exit_code == 3

    asyncio.run(run())


def test_process_cannot_remove_live():
    svc = ProcessService(EventHub())
    info_type = type(svc.tasks)  # sanity
    del info_type
    # inject a live task directly
    from forge_gui_server.services.procs import TaskInfo
    svc.tasks["t1"] = TaskInfo(id="t1", name="x", command=[], status="running")
    assert not svc.remove("t1")

# â”€â”€ AgentService loop binding â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_agent_bind_loop_keeps_loop_method():
    """bind_loop must not shadow the _loop drive method (regression:
    self._loop was used for both the asyncio loop and the agent loop,
    causing 'ProactorEventLoop object is not callable')."""
    from forge_gui_server.services.agent_loop import AgentRun, AgentService
    svc = AgentService(EventHub(), runtime=object())
    loop = asyncio.new_event_loop()
    try:
        svc.bind_loop(loop)
        assert svc._aio_loop is loop
        assert callable(svc._loop)
        assert not isinstance(svc._loop, asyncio.AbstractEventLoop)
    finally:
        loop.close()


def test_agent_drive_completes_stubbed():
    from forge_gui_server.services.agent_loop import AgentRun, AgentService
    svc = AgentService(EventHub(), runtime=object())
    svc._loop = lambda r: {"content": "ok", "rounds": 1, "tool_calls": [],
                           "sandbox": {}, "cancelled": False}
    run = AgentRun("a1", object(), "task", ".")
    svc._drive(run)
    assert run.status == "done"
    assert run.result["content"] == "ok"
    assert run.result["elapsed_s"] >= 0


# â”€â”€ Agent auto-stop / steering / workspace â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class _StubRT:
    info: dict = {}


class _StubStrikes:
    def summary(self):
        return {}


class _StubHarness:
    def tool_defs(self):
        return []
    def execute_calls(self, calls):
        return [{"ok": True, "result": {"text": "ok"}, "name": "x",
                 "elapsed_s": 0.0} for _ in calls]
    def summary(self):
        return {}
    @property
    def strikes(self):
        return _StubStrikes()


def test_agent_auto_mode_stops_when_model_done():
    """max_rounds=None â†’ run ends on the first no-tool-call round instead
    of burning a fixed round budget."""
    from forge_gui_server.services.agent_loop import AgentRun, AgentService
    svc = AgentService(EventHub(), runtime=_StubRT())
    svc._generate = lambda r, p: "all done - no more tools needed"
    run = AgentRun("a1", object(), "task", ".", max_rounds=None,
                   tool_harness=_StubHarness())
    out = svc._loop(run)
    assert out["rounds"] == 1
    assert out["content"] == "all done - no more tools needed"


def test_agent_fixed_rounds_caps():
    from forge_gui_server.services.agent_loop import AgentRun, AgentService
    svc = AgentService(EventHub(), runtime=_StubRT())
    svc._generate = lambda r, p: (
        '<tool_call>\n{"name": "x", "arguments": {}}\n</tool_call>')
    run = AgentRun("a2", object(), "task", ".", max_rounds=3,
                   tool_harness=_StubHarness())
    out = svc._loop(run)
    assert out["rounds"] == 3
    assert len(out["tool_calls"]) == 3


def test_agent_steer_and_detail():
    from forge_gui_server.services.agent_loop import AgentRun, AgentService
    svc = AgentService(EventHub(), runtime=object())
    run = AgentRun("a3", object(), "task", ".", tool_harness=_StubHarness())
    svc.runs["a3"] = run
    assert svc.steer("a3", "use the tmp dir instead")
    assert run._steer.get_nowait() == "use the tmp dir instead"
    assert run.events[-1]["kind"] == "user_message"
    d = svc.detail("a3")
    assert d["run"]["run_id"] == "a3"
    assert d["events"][-1]["kind"] == "user_message"
    assert svc.steer("a4", "nope") is False


def test_agent_workspace_projects_dir(tmp_path, monkeypatch):
    from forge_gui_server.routes import chat as chat_routes
    from forge_gui.api import status_reader
    monkeypatch.setattr(status_reader, "project_root", lambda: tmp_path)
    body = chat_routes.AgentStartRequest(task="Build a Todo CLI App")
    ws, project = chat_routes._agent_workspace(body)
    assert project == "build-a-todo-cli-app"
    assert ws.endswith("ForgeAI_Projects" + "\\" + project) or \
        ws.endswith("ForgeAI_Projects/" + project)
    assert (tmp_path / "ForgeAI_Projects" / project).is_dir()
    # explicit workspace escape hatch wins
    body2 = chat_routes.AgentStartRequest(
        task="x", workspace=str(tmp_path), project="named")
    assert chat_routes._agent_workspace(body2) == (str(tmp_path), "named")


# ── Self-play live bridge (events.jsonl → hub → /ws) ────────────────

def _watch_fixture(tmp_path):
    """Services stand-in wired to a tmp-dir EventsReader + real hub."""
    from types import SimpleNamespace
    from forge_gui.api.events_reader import EventsReader
    from forge_gui_server.deps import Services
    sp = tmp_path / "self_play"
    sp.mkdir()
    return (sp, SimpleNamespace(
        events=EventsReader(search_dirs=[tmp_path], rescan_interval=0),
        hub=EventHub()), Services._selfplay_watch)


def _drain(q, timeout=4.0):
    """Collect queued events until `timeout` elapses with none new."""
    out = []
    end = time.time() + timeout
    while time.time() < end:
        try:
            out.append(q.get_nowait())
        except asyncio.QueueEmpty:
            time.sleep(0.05)
    return out


def test_selfplay_watch_publishes_events(tmp_path):
    """Appended events.jsonl lines are republished once each, in order."""
    sp, svc, watch = _watch_fixture(tmp_path)
    q = svc.hub.subscribe(replay=False)

    async def run():
        t = asyncio.create_task(watch(svc))
        await asyncio.sleep(0.6)  # let the watcher discover the file
        ev_file = sp / "events.jsonl"
        with open(ev_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": 1, "kind": "task_done",
                                "success": True}) + "\n")
            f.write(json.dumps({"ts": 2, "kind": "phase",
                                "phase": "gen"}) + "\n")
        for _ in range(40):
            if q.qsize() >= 2:
                break
            await asyncio.sleep(0.05)
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass

    asyncio.run(run())
    got = [e for e in _drain(q, timeout=0.3) if e["type"] == "selfplay_event"]
    kinds = [e["payload"]["kind"] for e in got]
    assert kinds == ["task_done", "phase"]


def test_selfplay_watch_status_push(tmp_path):
    """status.json changes publish selfplay_status with heartbeat fields."""
    sp, svc, watch = _watch_fixture(tmp_path)
    (sp / "heartbeat.json").write_text(json.dumps({"ts": time.time()}))
    (sp / "status.json").write_text(json.dumps(
        {"status": "running", "phase": "gen"}))
    q = svc.hub.subscribe(replay=False)

    async def run():
        t = asyncio.create_task(watch(svc))
        for _ in range(40):
            if not q.empty():
                break
            await asyncio.sleep(0.05)
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass

    asyncio.run(run())
    got = [e for e in _drain(q, timeout=0.3)
           if e["type"] == "selfplay_status"]
    assert got
    p = got[-1]["payload"]
    assert p["status"]["phase"] == "gen"
    assert "heartbeat_age_s" in p and "heartbeat_stalled" in p


def test_selfplay_watch_survives_truncation(tmp_path):
    """New-run truncation must not wedge the watcher or re-emit stale
    events at garbage offsets."""
    sp, svc, watch = _watch_fixture(tmp_path)
    ev_file = sp / "events.jsonl"
    ev_file.write_text(json.dumps({"ts": 0, "kind": "old"}) + "\n" * 3)
    q = svc.hub.subscribe(replay=False)

    async def run():
        t = asyncio.create_task(watch(svc))
        await asyncio.sleep(0.6)
        # simulate run start: file wiped, then fresh events — each write
        # gets its own watcher tick (0.4s poll) so Windows stat caching
        # can't merge them into one read window
        ev_file.write_text(json.dumps({"ts": 5, "kind": "phase",
                                       "phase": "startup"}) + "\n")
        for _ in range(40):
            if q.qsize() >= 2:  # backfill "old" + post-truncation "phase"
                break
            await asyncio.sleep(0.05)
        with open(ev_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": 6, "kind": "task_start",
                                "idx": 0}) + "\n")
        for _ in range(40):
            if q.qsize() >= 3:
                break
            await asyncio.sleep(0.05)
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass

    asyncio.run(run())
    kinds = [e["payload"]["kind"] for e in _drain(q, timeout=0.3)
             if e["type"] == "selfplay_event"]
    # backfill "old" + the two fresh records — no crash, no stalls
    assert "task_start" in kinds and "phase" in kinds


def test_pipe_reset_filter_swallows_10054():
    """Windows subprocess pipe teardown logs a cosmetic
    ConnectionResetError via the loop's default exception handler —
    the filter must swallow it and pass real errors through."""
    from forge_gui_server.deps import Services
    loop = asyncio.new_event_loop()
    seen = []
    loop.set_exception_handler(lambda l, ctx: seen.append(ctx))
    Services._install_pipe_reset_filter(loop)
    handler = loop.get_exception_handler()
    if sys.platform == "win32":
        # pipe-reset noise: swallowed (debug-logged, not forwarded)
        handler(loop, {"exception": ConnectionResetError(10054),
                       "message": "pipe lost"})
        assert seen == []
        # real errors still reach the previous handler
        handler(loop, {"exception": RuntimeError("boom")})
        assert len(seen) == 1
    else:
        # non-Windows: filter is a no-op, handler unchanged
        handler(loop, {"exception": ConnectionResetError(10054)})
        assert len(seen) == 1
    loop.close()


