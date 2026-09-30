"""Service registry — constructs and wires all backend services.

Mirrors the MainWindow backend graph from forge_gui/app.py: eager
services (gpu, status, models index, procs, engine runtime) plus lazy
stores (chat, lorebook, mcp, web tools, lora training, tool harness).
"""
from __future__ import annotations

import asyncio
import logging
import os

from forge_gui.api.chat_store import ChatStore
from forge_gui.api.events_reader import EventsReader
from forge_gui.api.log_tailer import LogTailer
from forge_gui.api.lorebook import Lorebook
from forge_gui.api.master_prompt import (
    get_default_prompt_for_config, generate_master_prompt)
from forge_gui.api.mcp_client import MCPManager
from forge_gui.api.models_index import ModelsIndex
from forge_gui.api.status_reader import StatusReader, project_root
from forge_gui.api.web_tools import WebTools

from .hub import EventHub
from .services.agent_loop import AgentService
from .services.boot import BootService
from .services.engine_rt import EngineService
from .services.gpu import GpuMonitor
from .services.lora import LoraService
from .services.procs import ProcessService
from .services.support import (
    ApprovalChannel, BackupService, LibraryService, LoraHarnessAdapter,
    SubAgentService, TimeService)

logger = logging.getLogger(__name__)


class Services:
    def __init__(self) -> None:
        self.hub = EventHub()
        self.approvals = ApprovalChannel(self.hub)

        # eager
        self.gpu = GpuMonitor(interval_s=2.0)
        self.status_reader = StatusReader()
        self.models_index = ModelsIndex()
        self.log_tailer = LogTailer()
        self.events = EventsReader()
        self.procs = ProcessService(self.hub)
        self.engine = EngineService(self.hub)
        self.lora = LoraService(self.hub, self.engine)
        self.boot = BootService(self.hub, self.engine)
        self.agent = AgentService(self.hub, self.engine)
        self.sub_agents = SubAgentService(self.hub, self.engine)
        self.timers = TimeService(self.hub)
        self.backups = BackupService(self.hub, self.approvals)
        self.library = LibraryService(self.hub, self.approvals)

        # lazy
        self._chat_store: ChatStore | None = None
        self._lorebook: Lorebook | None = None
        self._mcp: MCPManager | None = None
        self._web_tools: WebTools | None = None
        self._lora_training = None
        self._lora_adapter: LoraHarnessAdapter | None = None

    # ── lazy stores ───────────────────────────────────────────────────
    @property
    def chat_store(self) -> ChatStore:
        if self._chat_store is None:
            self._chat_store = ChatStore()
        return self._chat_store

    @property
    def lorebook(self) -> Lorebook:
        if self._lorebook is None:
            self._lorebook = Lorebook()
        return self._lorebook

    @property
    def mcp(self) -> MCPManager:
        if self._mcp is None:
            self._mcp = MCPManager()
        return self._mcp

    @property
    def web_tools(self) -> WebTools:
        if self._web_tools is None:
            self._web_tools = WebTools(enabled=True)
        return self._web_tools

    @property
    def lora_training(self):
        if self._lora_training is None:
            from forge_gui.api.lora_training_trigger import (
                LoraTrainingTrigger)
            self._lora_training = LoraTrainingTrigger(
                proc_mgr=self.procs, chat_store=self.chat_store,
                checkpoint="research/checkpoints/ForgeLM_V2.safetensors")
        return self._lora_training

    def lora_adapter(self, loop) -> LoraHarnessAdapter:
        if self._lora_adapter is None:
            self._lora_adapter = LoraHarnessAdapter(
                self.lora, self.engine, loop)
        return self._lora_adapter

    # ── tool harness factory ──────────────────────────────────────────
    def make_harness(self, workspace: str | None = None,
                     read_only: bool = False):
        from forge_gui.api.tool_harness import ToolHarness
        loop = asyncio.get_event_loop()
        return ToolHarness(
            workspace=workspace or str(project_root()),
            lorebook=self.lorebook,
            lora_harness=self.lora_adapter(loop),
            mcp_manager=self.mcp,
            lora_training=self.lora_training,
            backup_manager=self.backups,
            sub_agent_manager=self.sub_agents,
            time_manager=self.timers,
            library_manager=self.library,
            web_tools=self.web_tools,
            read_only=read_only,
            enable_safety=True,
        )

    # ── lifecycle ─────────────────────────────────────────────────────
    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._install_pipe_reset_filter(loop)
        self.hub.bind_loop(loop)
        self.procs.bind_loop(loop)
        self.engine.bind_loop(loop)
        self.agent.bind_loop(loop)
        self.gpu.start(self.hub)
        self.agent.set_harness_factory(
            lambda workspace=None: self.make_harness(workspace))
        self._selfplay_task = asyncio.create_task(self._selfplay_watch())
        self._preload_engine()

    async def _selfplay_watch(self) -> None:
        """Tail self-play telemetry files -> push onto the event hub.

        The self-play loop runs as a subprocess writing status.json /
        events.jsonl (EventsReader already does offset-tracked tailing).
        Republishing onto /ws gives the SelfPlay page sub-second updates
        instead of 3s REST polling; the REST endpoint stays for initial
        hydration.  Runs cheap: ~400ms cadence, stat+read of appended
        bytes only; publishes nothing while a run is idle."""
        loop = asyncio.get_running_loop()
        last_status: dict | None = None
        while True:
            try:
                new_events = await loop.run_in_executor(
                    None, self.events.poll)
                for ev in new_events:
                    self.hub.publish("selfplay_event", ev)
                status = await loop.run_in_executor(
                    None, self.events.latest_status)
                if status is not None and status != last_status:
                    last_status = status
                    self.hub.publish("selfplay_status", {
                        "status": status,
                        "heartbeat_age_s": self.events.heartbeat_age(),
                        "heartbeat_stalled": self.events.heartbeat_stalled(),
                    })
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("selfplay watch iteration failed",
                             exc_info=True)
            await asyncio.sleep(0.4)

    @staticmethod
    def _install_pipe_reset_filter(loop: asyncio.AbstractEventLoop) -> None:
        """Suppress WinError 10054 noise from subprocess pipe teardown.

        On Windows, when a ProcessService child exits (or is killed) the
        remote end of the stdout pipe is already gone; the proactor's
        ``_call_connection_lost`` then calls ``sock.shutdown`` on a dead
        socket and the loop's default handler logs a ConnectionResetError
        that looks like a crash but is harmless.  Swallow just that class
        of callback error; everything else goes to the previous/default
        handler unchanged."""
        if os.name != "nt":
            return
        prev = loop.get_exception_handler()

        def _handler(l: asyncio.AbstractEventLoop, context: dict) -> None:
            exc = context.get("exception")
            if isinstance(exc, (ConnectionResetError,
                                ConnectionAbortedError, BrokenPipeError)):
                logger.debug("subprocess pipe reset on child exit "
                             "(harmless): %s", exc)
                return
            if prev is not None:
                prev(l, context)
            else:
                l.default_exception_handler(context)

        loop.set_exception_handler(_handler)

    def _preload_engine(self) -> None:
        """Kick off the resident model load in the background at startup.

        The load runs on the engine executor thread (non-blocking), so the
        UI serves immediately and engine state/progress streams over /ws —
        by the time the user opens Chat the model is typically ready.

        Env:
          FORGE_GUI_NO_PRELOAD=1        disable preload
          FORGE_GUI_PRELOAD=<path>      checkpoint (default: ForgeLM_V2)
          FORGE_GUI_PRELOAD_CONFIG=<s>  config preset (default: forgelm_v2)
        """
        if os.environ.get("FORGE_GUI_NO_PRELOAD", "").strip().lower() in (
                "1", "true", "yes"):
            return
        ckpt = os.environ.get("FORGE_GUI_PRELOAD", "")
        cfg = os.environ.get("FORGE_GUI_PRELOAD_CONFIG", "forgelm_v2")
        resolved = self.engine._resolve_ckpt(ckpt)
        if not os.path.isfile(resolved):
            logger.info("engine preload skipped: no checkpoint at %s",
                        resolved)
            return
        logger.info("preloading engine: %s (%s)", resolved, cfg)
        self.engine.load(checkpoint=ckpt, config_name=cfg)

    async def stop(self) -> None:
        t = getattr(self, "_selfplay_task", None)
        if t is not None:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
        self.backups.stop()
        self.timers.shutdown()
        self.sub_agents.shutdown()
        await self.gpu.stop()
        await self.procs.shutdown()
        await self.engine.shutdown()


services = Services()
