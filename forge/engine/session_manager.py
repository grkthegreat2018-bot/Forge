"""Session manager + request batching queue for concurrent multi-task generation.

Layer map: this file = server-side conversation history + BatchQueue.
Per-session KV reuse lives in session_cache.py + the engine_sessions.py
mixin (different layer despite the similar name).

Provides:
  - SessionManager: per-task conversation history + KV cache, LRU eviction
  - BatchQueue: collects concurrent requests for a configurable window
    (default 50ms), then dispatches them as a single batched forward pass
    via BatchedDecoding. Non-streaming and streaming both supported.

Design:
  - Each task has a unique ID, conversation history, and optional KV cache
    snapshot (for continuation without re-prefilling the full history).
  - The BatchQueue runs a background dispatcher thread that wakes every
    `batch_window_ms`, collects all pending requests, and runs them as a
    single batch. Results are delivered to per-request futures.
  - Streaming: individual token chunks are yielded to each requester as
    they emerge from the batched decode loop.

Usage:
    from forge.engine.session_manager import SessionManager, BatchQueue

    sm = SessionManager(max_sessions=64, max_history_tokens=8192)
    bq = BatchQueue(registry, session_manager=sm, batch_window_ms=50)

    # Create a task
    task_id = sm.create_task(model_id="forgelm-v2-jamba", system_prompt="You are a coder.")

    # Submit generation (non-blocking)
    future = bq.submit(task_id, "Write a fibonacci function", max_tokens=256)
    result = future.result()  # blocks until this request's batch completes

    # Continue the task
    future2 = bq.submit(task_id, "Now make it recursive", max_tokens=128)
"""
import asyncio
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

import torch

from forge.engine.batched_decoding import BatchedDecoding

# ── Task Boot Config ─────────────────────────────────────────────────────────

@dataclass
class TaskBootConfig:
    """Per-task boot-time engine configuration.

    Allows each task to independently specify engine parameters that
    traditionally are fixed at model load time. When a task's config
    differs from the engine's current state, the engine is reconfigured
    before generation (via ForgeEngine.activate()).

    This is a superset of what LM Studio exposes — LM Studio only allows
    global model settings, not per-conversation boot params.
    """
    kv_cache: str = "standard"  # standard, paged, hadamard_int4, snapkv, snapkv_4bit, etc.
    decoding: str = "standard"  # standard, speculative, batched, mtp_selfspec, etc.
    quantize: str | None = None  # None, int8, int4, fp8
    acceleration: str | None = None  # None, cuda_graph, airllm_streaming
    kv_cache_tokens: int | None = None  # Limit KV cache allocation (saves VRAM)
    kv_bits: int = 4  # 4 or 8, for KV cache quantization
    mrl_keep_ratio: float | None = None  # Truncate to fraction of dims
    use_v0_warm: bool = False  # V0 warm-start for KV cache
    use_progressive_kv: bool = False  # Progressive KV (anchor + residual)
    use_compile: bool = False  # torch.compile the model
    use_triton_conv: bool = False  # Fused Triton conv kernel
    use_prefix_cache: bool = False  # Cache KV for repeated prefixes
    use_spec_attn: bool = False  # L1 Speculative Attention
    warmup: bool = True  # Pre-run dummy token to init CUDA kernels

    def differs_from(self, engine) -> bool:
        """Check if this config differs from the engine's current state."""
        if self.kv_cache != "standard" and (
            engine.kv_cache is None or
            getattr(engine.kv_cache, 'name', 'standard') != self.kv_cache):
            return True
        if self.decoding != "standard" and engine.decoding.name != self.decoding:
            return True
        if self.quantize != engine.quantize:
            return True
        if self.acceleration != engine.acceleration:
            return True
        return False

    def apply_to(self, engine):
        """Apply this config to a ForgeEngine via activate()."""
        # Serialise against in-flight generation: activate() swaps
        # decoding/kv/quant strategies mid-request otherwise.
        with engine._gen_lock:
            engine.activate(
                kv_cache=self.kv_cache,
                decoding=self.decoding,
                quantize=self.quantize,
                acceleration=self.acceleration,
                kv_cache_tokens=self.kv_cache_tokens,
                kv_bits=self.kv_bits,
                mrl_keep_ratio=self.mrl_keep_ratio,
                use_v0_warm=self.use_v0_warm,
                use_progressive_kv=self.use_progressive_kv,
                use_compile=self.use_compile,
                use_triton_conv=self.use_triton_conv,
                use_prefix_cache=self.use_prefix_cache,
                use_spec_attn=self.use_spec_attn,
                warmup=self.warmup,
            )

    def to_dict(self) -> dict:
        return {
            "kv_cache": self.kv_cache,
            "decoding": self.decoding,
            "quantize": self.quantize,
            "acceleration": self.acceleration,
            "kv_cache_tokens": self.kv_cache_tokens,
            "kv_bits": self.kv_bits,
            "mrl_keep_ratio": self.mrl_keep_ratio,
            "use_v0_warm": self.use_v0_warm,
            "use_progressive_kv": self.use_progressive_kv,
            "use_compile": self.use_compile,
            "use_triton_conv": self.use_triton_conv,
            "use_prefix_cache": self.use_prefix_cache,
            "use_spec_attn": self.use_spec_attn,
            "warmup": self.warmup,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "TaskBootConfig":
        if d is None:
            return cls()
        # Filter to known fields
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


# ── Session Manager ──────────────────────────────────────────────────────────

@dataclass
class TaskSession:
    """A single conversation/task context."""
    task_id: str
    model_id: str
    system_prompt: str
    seed: int | None = None  # default seed for all generations in this task
    boot_config: TaskBootConfig = field(default_factory=TaskBootConfig)
    history: list[dict] = field(default_factory=list)
    # history: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    total_tokens: int = 0
    # Optional: cached KV state for continuation (not yet populated)
    kv_cache: Any = None
    kv_seq_len: int = 0


class SessionManager:
    """Per-task conversation context with LRU eviction.

    Each task has:
      - Unique task ID (UUID)
      - Conversation history (user/assistant messages)
      - Model ID (which model to use)
      - Optional system prompt
      - Token budget tracking

    Sessions are evicted LRU when max_sessions is exceeded.
    """

    def __init__(self, max_sessions: int = 64, max_history_tokens: int = 8192):
        self.max_sessions = max_sessions
        self.max_history_tokens = max_history_tokens
        self._sessions: OrderedDict[str, TaskSession] = OrderedDict()
        self._lock = threading.RLock()

    def create_task(self, model_id: str, system_prompt: str = "",
                    seed: int | None = None,
                    boot_config: TaskBootConfig | dict | None = None,
                    task_id: str | None = None) -> str:
        """Create a new task session. Returns the task ID.

        Args:
            model_id: Which model to use for this task.
            system_prompt: System prompt prepended to all generations.
            seed: Default random seed for this task's generations.
                None = random (non-deterministic). Set to an int for
                reproducible generation across runs with the same prompt.
            boot_config: Per-task engine boot parameters (KV cache strategy,
                decoding strategy, quantization, acceleration, etc.).
                Accepts a TaskBootConfig or a dict. None = engine defaults.
            task_id: Optional explicit task ID (auto-generated if None).
        """
        with self._lock:
            tid = task_id or f"task_{uuid.uuid4().hex[:12]}"
            if isinstance(boot_config, dict):
                boot_config = TaskBootConfig.from_dict(boot_config)
            elif boot_config is None:
                boot_config = TaskBootConfig()
            session = TaskSession(
                task_id=tid,
                model_id=model_id,
                system_prompt=system_prompt,
                seed=seed,
                boot_config=boot_config,
            )
            self._sessions[tid] = session
            self._evict_if_needed()
            return tid

    def update_boot_config(self, task_id: str,
                           boot_config: TaskBootConfig | dict) -> bool:
        """Update a task's boot configuration. Returns True if task existed.

        The new config will be applied on the next generation request.
        """
        with self._lock:
            session = self._sessions.get(task_id)
            if session is None:
                return False
            if isinstance(boot_config, dict):
                boot_config = TaskBootConfig.from_dict(boot_config)
            session.boot_config = boot_config
            session.last_used = time.time()
            self._sessions.move_to_end(task_id)
            return True

    def get_task(self, task_id: str) -> TaskSession | None:
        """Get a task session by ID. Returns None if not found."""
        with self._lock:
            session = self._sessions.get(task_id)
            if session is None:
                return None
            # Move to end (most recently used)
            self._sessions.move_to_end(task_id)
            session.last_used = time.time()
            return session

    def append_message(self, task_id: str, role: str, content: str):
        """Append a message to the task's conversation history.

        Trims oldest messages when total token estimate exceeds
        max_history_tokens to prevent unbounded memory growth in
        long-running sessions.
        """
        with self._lock:
            session = self._sessions.get(task_id)
            if session is None:
                raise KeyError(f"Task '{task_id}' not found")
            session.history.append({"role": role, "content": content})
            # Trim oldest messages if token budget exceeded (~4 chars/token)
            char_budget = self.max_history_tokens * 4
            while len(session.history) > 2:
                total_chars = sum(len(m["content"]) for m in session.history)
                if total_chars <= char_budget:
                    break
                # Keep at least the system + last user message
                session.history.pop(0)
            session.last_used = time.time()
            self._sessions.move_to_end(task_id)

    def get_history(self, task_id: str) -> list[dict]:
        """Get the full conversation history for a task."""
        with self._lock:
            session = self._sessions.get(task_id)
            if session is None:
                raise KeyError(f"Task '{task_id}' not found")
            return list(session.history)

    def delete_task(self, task_id: str) -> bool:
        """Delete a task session. Returns True if it existed."""
        with self._lock:
            if task_id in self._sessions:
                del self._sessions[task_id]
                return True
            return False

    def list_tasks(self) -> list[dict]:
        """List all active tasks with metadata."""
        with self._lock:
            now = time.time()
            return [
                {
                    "task_id": s.task_id,
                    "model_id": s.model_id,
                    "messages": len(s.history),
                    "total_tokens": s.total_tokens,
                    "age_s": round(now - s.created_at, 1),
                    "last_used_s": round(now - s.last_used, 1),
                    "seed": s.seed,
                    "boot_config": s.boot_config.to_dict(),
                }
                for s in self._sessions.values()
            ]

    def build_prompt(self, task_id: str, new_user_message: str,
                    render_fn=None) -> str:
        """Build the full prompt for a continuation request.

        Args:
            task_id: The task to continue.
            new_user_message: The new user message to append.
            render_fn: Optional function(messages, system_prompt) -> str.
                If None, uses a simple chat template.

        Returns:
            The rendered prompt string.
        """
        with self._lock:
            session = self._sessions.get(task_id)
            if session is None:
                raise KeyError(f"Task '{task_id}' not found")

            # Build messages list
            messages = []
            if session.system_prompt:
                messages.append({"role": "system", "content": session.system_prompt})
            messages.extend(session.history)
            messages.append({"role": "user", "content": new_user_message})

            # Append to history
            session.history.append({"role": "user", "content": new_user_message})
            session.last_used = time.time()
            self._sessions.move_to_end(task_id)

        if render_fn:
            return render_fn(messages)
        # Default: simple chat template
        return _default_render(messages)

    def _evict_if_needed(self):
        """Evict oldest sessions if over max_sessions."""
        while len(self._sessions) > self.max_sessions:
            tid, session = self._sessions.popitem(last=False)
            # Could log eviction here


def _default_render(messages: list[dict]) -> str:
    """Simple chat template: <|startoftext|>system\n...user\n...assistant\n"""
    from forge.self_play.discovery.qwen_adapter import qwen_render_messages
    return qwen_render_messages(messages, add_generation_prompt=True)


# ── Batch Queue ──────────────────────────────────────────────────────────────

@dataclass
class PendingRequest:
    """A request waiting to be batched."""
    task_id: str | None  # None for stateless requests
    model_id: str
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    seed: int | None
    stop: list[str] | None
    stream: bool
    future: Future
    # For streaming: an asyncio.Queue to push chunks to, and the event
    # loop it belongs to (captured at submit() — the dispatcher thread
    # has no loop, so asyncio.get_event_loop() would crash there).
    stream_queue: asyncio.Queue | None = None
    loop: asyncio.AbstractEventLoop | None = None
    created_at: float = field(default_factory=time.time)


class BatchQueue:
    """Collects concurrent requests and dispatches them as batched forward passes.

    A background dispatcher thread wakes every `batch_window_ms` milliseconds,
    collects all pending requests for each model, and runs them as a single
    batched generation via BatchedDecoding.

    For streaming requests, token chunks are pushed to per-request asyncio.Queue
    objects, which the FastAPI endpoint reads via SSE.
    """

    def __init__(self, registry, session_manager: SessionManager | None = None,
                 batch_window_ms: int = 52, max_batch_size: int = 15,
                 use_feather_scheduler: bool = False,
                 continuous_batching: bool = False):
        self.registry = registry
        self.session_manager = session_manager or SessionManager()
        self.batch_window = batch_window_ms / 1000.0
        self.max_batch_size = max_batch_size
        self.continuous_batching = continuous_batching

        self._pending: list[PendingRequest] = []
        self._lock = threading.Lock()

        # Feather prefix-homogeneity scheduler (2-10x for prefix-sharing workloads)
        self._feather = None
        if use_feather_scheduler:
            from forge.engine.scheduler.feather_scheduler import FeatherScheduler
            self._feather = FeatherScheduler(
                max_batch_size=max_batch_size,
                homogeneity_threshold=0.5)
        self._cv = threading.Condition(self._lock)
        self._running = False
        self._dispatcher_thread: threading.Thread | None = None

    def start(self):
        """Start the background dispatcher thread."""
        if self._running:
            return
        self._running = True
        self._dispatcher_thread = threading.Thread(
            target=self._dispatch_loop, daemon=True)
        self._dispatcher_thread.start()

    def stop(self):
        """Stop the dispatcher thread."""
        self._running = False
        with self._cv:
            self._cv.notify_all()
        if self._dispatcher_thread:
            self._dispatcher_thread.join(timeout=5.0)

    def submit(self, task_id: str | None, model_id: str, prompt: str,
               max_tokens: int = 256, temperature: float = 0.0,
               top_p: float = 1.0, top_k: int = 80,
               repetition_penalty: float = 1.05,
               seed: int | None = None,
               stop: list[str] | None = None,
               stream: bool = False,
               stream_queue: asyncio.Queue | None = None) -> Future:
        """Submit a request to the batch queue.

        Returns a Future that will be resolved with the generated text
        (for non-streaming) or None (for streaming — chunks go to stream_queue).

        Args:
            seed: Random seed for reproducible generation. None = random.
                Same seed + same prompt = same output (when temperature > 0).
            top_k: Top-k sampling (0 = disabled, 80 = LFM2.5 default).
            repetition_penalty: Repetition penalty (1.0 = disabled, 1.05 = default).
            stop: Stop sequences — generation halts when any is encountered.
            stream_queue: asyncio.Queue for streaming chunks. Must belong
                to the event loop the caller is running on — the loop is
                captured here because the dispatcher thread cannot call
                ``asyncio.get_event_loop()``.
        """
        future: Future = Future()
        loop = None
        if stream and stream_queue is not None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # Caller is not inside a loop — leave None; the dispatcher
                # will fall back to resolving the future directly.
                pass
        req = PendingRequest(
            task_id=task_id,
            model_id=model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,
            stop=stop,
            stream=stream,
            future=future,
            stream_queue=stream_queue,
            loop=loop,
        )
        with self._cv:
            self._pending.append(req)
            self._cv.notify()
        return future

    def _dispatch_loop(self):
        """Main dispatcher loop: collect + batch + dispatch.

        In continuous_batching mode: dispatches immediately when any request
        is available (no fixed window wait), and new requests can join the
        next iteration step. This is iteration-level batching like vLLM/SGLang,
        not request-level batching.

        In default mode: waits up to batch_window (52ms) for requests to
        accumulate, then dispatches the full batch.
        """
        while self._running:
            with self._cv:
                if self.continuous_batching:
                    # Continuous: wait only until first request arrives
                    self._cv.wait_for(
                        lambda: self._pending or not self._running)
                else:
                    # Fixed window: wait for window or requests
                    self._cv.wait_for(
                        lambda: self._pending or not self._running,
                        timeout=self.batch_window)
                if not self._running:
                    break
                if not self._pending:
                    continue

                # Collect all pending requests (up to max_batch_size per model)
                batch = self._pending[:self.max_batch_size]
                self._pending = self._pending[len(batch):]

            # Group by (model_id, boot config) — requests with different
            # per-task boot configs CANNOT share one batched pass, since
            # the engine reconfigures (KV cache, decoding, quant) before
            # generating.  Grouping instead of taking request[0]'s config
            # prevents silent cross-config contamination.
            by_model: dict[tuple, list[PendingRequest]] = {}
            for req in batch:
                by_model.setdefault(
                    (req.model_id, self._boot_key(req)), []).append(req)

            # Dispatch each (model, config) group's batch
            for (model_id, _bk), reqs in by_model.items():
                try:
                    self._run_batch(model_id, reqs)
                except Exception as e:
                    for req in reqs:
                        if not req.future.done():
                            req.future.set_exception(e)

    def _boot_key(self, req: PendingRequest) -> tuple:
        """Hashable boot-config key for batching.  Requests with equal
        keys are safe to run in the same batched pass."""
        if req.task_id and self.session_manager:
            session = self.session_manager.get_task(req.task_id)
            if session and session.boot_config:
                return tuple(sorted(session.boot_config.to_dict().items()))
        return ()

    @staticmethod
    def _push_stream(req: PendingRequest, item) -> bool:
        """Push an item to a streaming request's asyncio queue.

        Returns False (caller should fall back to the future) when the
        request has no captured event loop.
        """
        if req.stream_queue is None or req.loop is None:
            return False
        asyncio.run_coroutine_threadsafe(
            req.stream_queue.put(item), req.loop)
        return True

    def _run_batch(self, model_id: str, requests: list[PendingRequest]):
        """Run a batch of requests through BatchedDecoding."""
        if not requests:
            return

        # Apply boot config from the first request's task (if it has one)
        # For batched requests, all must use the same engine config — we use
        # the first task's config. Mixed boot configs in one batch fall back
        # to single-request mode (handled by B==1 path below).
        boot_config = None
        if requests[0].task_id and self.session_manager:
            session = self.session_manager.get_task(requests[0].task_id)
            if session and session.boot_config:
                boot_config = session.boot_config

        # Get the engine and apply boot config if needed
        with self.registry._lock:
            if model_id not in self.registry._entries:
                raise KeyError(f"Model '{model_id}' not registered")
            self.registry._ensure_awake(model_id)
            entry = self.registry._entries[model_id]
            entry.last_used = time.time()
            engine = entry.engine

        # Apply boot config if it differs from engine's current state
        if boot_config and boot_config.differs_from(engine):
            boot_config.apply_to(engine)

        # Single request: use the normal generate path (faster for B=1)
        if len(requests) == 1:
            req = requests[0]
            try:
                # Resolve seed: request seed, else task's default seed
                effective_seed = req.seed
                if effective_seed is None and req.task_id and self.session_manager:
                    session = self.session_manager.get_task(req.task_id)
                    if session and session.seed is not None:
                        effective_seed = session.seed
                if req.stream:
                    run = lambda: self._run_single_stream(req)
                else:
                    def run():
                        output = self.registry.generate(
                            req.model_id, req.prompt,
                            max_new_tokens=req.max_tokens,
                            temperature=req.temperature,
                            top_p=req.top_p,
                        )
                        if req.task_id and self.session_manager:
                            self.session_manager.append_message(
                                req.task_id, "assistant", output)
                        req.future.set_result(output)
                # Seed scoped to THIS request — a bare torch.manual_seed
                # would leak into every other in-flight request's sampling.
                if effective_seed is not None:
                    devices = ([engine.device]
                               if engine.device.type == "cuda" else [])
                    with torch.random.fork_rng(devices=devices):
                        torch.manual_seed(effective_seed)
                        run()
                else:
                    run()
            except Exception as e:
                req.future.set_exception(e)
            return

        # Multiple requests: batched generation
        try:
            self._run_batched(model_id, requests)
        except Exception as e:
            for req in requests:
                if not req.future.done():
                    req.future.set_exception(e)

    def _run_batched(self, model_id: str, requests: list[PendingRequest]):
        """Run multiple requests as a single batched forward pass.

        Uses Feather prefix-homogeneity scheduling when available: groups
        requests by shared prefix for better KV cache locality (2-10x
        throughput for prefix-sharing workloads like multi-turn chat).
        """
        # Get the engine for this model
        with self.registry._lock:
            if model_id not in self.registry._entries:
                raise KeyError(f"Model '{model_id}' not registered")
            self.registry._ensure_awake(model_id)
            entry = self.registry._entries[model_id]
            entry.last_used = time.time()
            engine = entry.engine

        model = engine.model
        tokenizer = engine.tokenizer
        device = engine.device

        # Tokenize all prompts
        prompt_ids_list = []
        max_tokens_list = []
        temps_list = []
        top_ps_list = []
        top_ks_list = []
        rep_penalties_list = []
        seeds_list = []
        stop_list = []

        for req in requests:
            ids = tokenizer(req.prompt, return_tensors="pt")
            if hasattr(ids, "to"):
                ids = ids.to(device)
            else:
                ids = {k: v.to(device) for k, v in ids.items()}
            input_ids = ids["input_ids"] if isinstance(ids, dict) else ids.input_ids
            # Chunked prefill: if the prompt is long and the engine has
            # chunked prefill enabled, use it to avoid blocking decode queue.
            if (hasattr(engine, '_chunked_prefill') and
                    engine._chunked_prefill is not None and
                    input_ids.shape[1] > 1024):
                from forge.engine.prefill import should_chunk
                if should_chunk(input_ids.shape[1]):
                    # For chunked prefill, we handle this request separately
                    # (not batched) — long prompts don't batch well anyway.
                    pass  # The generate method handles this via ForgeEngine
            prompt_ids_list.append(input_ids)
            max_tokens_list.append(req.max_tokens)
            temps_list.append(req.temperature)
            top_ps_list.append(req.top_p)
            top_ks_list.append(req.top_k)
            rep_penalties_list.append(req.repetition_penalty)
            # Resolve seed: request seed, else task's default seed
            eff_seed = req.seed
            if eff_seed is None and req.task_id and self.session_manager:
                session = self.session_manager.get_task(req.task_id)
                if session and session.seed is not None:
                    eff_seed = session.seed
            seeds_list.append(eff_seed)
            stop_list.append(req.stop)

        # Memory models (FluxLM): BatchedDecoding assumes KV-cache forward
        # semantics and would interleave unrelated prompts into one memory
        # stream — route through the engine's stream-pool instead (shared
        # read-only memory, per-sequence context, zero mutation).
        from forge.engine.engine_common import _is_memory_model
        if _is_memory_model(model):
            outs = engine.generate_batch(
                [r.prompt for r in requests],
                max_new_tokens=max(max_tokens_list),
                temperatures=temps_list, top_ps=top_ps_list,
                top_ks=top_ks_list,
                repetition_penalties=rep_penalties_list,
                seeds=seeds_list, stops=stop_list,
                skip_special_tokens=True)
            for req, text in zip(requests, outs):
                if req.task_id and self.session_manager:
                    self.session_manager.append_message(
                        req.task_id, "assistant", text)
                if req.stream and req.stream_queue is not None:
                    if self._push_stream(req, text):
                        self._push_stream(req, None)
                    else:
                        req.future.set_result(text)
                else:
                    req.future.set_result(text)
            with self.registry._lock:
                entry.generation_count += len(requests)
                entry.total_tokens += sum(
                    len(tokenizer.encode(t)) for t in outs)
            return

        # Run batched decoding with per-sequence settings.  Hold the
        # engine's generation lock — BatchedDecoding drives model.forward
        # directly and would otherwise interleave with in-flight requests
        # on the shared conv/recurrent state.  Route through the engine's
        # OOM recovery so one OOM doesn't fail the whole batch.
        decoder = BatchedDecoding(
            eos_token_id=tokenizer.eos_token_id or 7,
            eos_token_ids=engine._eos_token_ids(),
        )

        def _decode():
            with engine._gen_lock:
                return decoder.generate_batch(
                    model,
                    prompt_ids_list,
                    max_tokens_list,
                    temps_list,
                    top_ps_list,
                    top_k_list=top_ks_list,
                    repetition_penalty_list=rep_penalties_list,
                    seed_list=seeds_list,
                    stop_list=stop_list,
                    tokenizer=tokenizer,
                )

        generated = engine._generate_with_oom_recovery(_decode)

        # Decode and deliver results
        for req, gen_ids in zip(requests, generated):
            gen_ids_1d = gen_ids[0] if gen_ids.dim() > 1 else gen_ids
            text = tokenizer.decode(gen_ids_1d, skip_special_tokens=True)

            # Record in session history
            if req.task_id and self.session_manager:
                self.session_manager.append_message(
                    req.task_id, "assistant", text)

            if req.stream and req.stream_queue is not None:
                # Push chunks to the stream queue on the loop captured at
                # submit() — get_event_loop() crashes in this thread.
                # For simplicity, push the whole text as one chunk
                # (true per-token streaming in batched mode would require
                #  modifying BatchedDecoding to yield per-step)
                if self._push_stream(req, text):
                    self._push_stream(req, None)  # sentinel
                else:
                    req.future.set_result(text)
            else:
                req.future.set_result(text)

        # Update stats
        with self.registry._lock:
            entry.generation_count += len(requests)
            entry.total_tokens += sum(
                g.shape[-1] for g in generated)

    def _run_single_stream(self, req: PendingRequest):
        """Stream a single request token-by-token."""
        accumulated = ""
        for chunk in self.registry.generate_stream(
            req.model_id, req.prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stop=req.stop,
        ):
            accumulated += chunk
            if req.stream_queue is not None:
                self._push_stream(req, chunk)

        # Record in session history
        if req.task_id and self.session_manager:
            self.session_manager.append_message(
                req.task_id, "assistant", accumulated)

        # Signal end of stream
        if req.stream_queue is not None:
            self._push_stream(req, None)

        req.future.set_result(accumulated)
