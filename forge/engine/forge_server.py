"""ForgeAI OpenAI-compatible HTTP server.

FastAPI server that wraps ModelRegistry with OpenAI-compatible endpoints.
Supports SSE streaming, model routing, sleep/wake management, tool calling,
Prometheus-compatible health checks, **concurrent task-based generation**
with session management, and **request batching** for multi-task throughput.

Surpasses LM Studio's proxy-only approach by serving models directly
with all ForgeAI optimizations (MTP, QuaRot, ProgressiveKV, batched decode,
session-based task continuity, etc.).

Usage:
    python forge/engine/forge_server.py --models forgelm-v2-jamba

    # Or programmatically:
    from forge.engine.forge_server import ForgeServer
    server = ForgeServer()
    server.register("forgelm-v10", checkpoint="...", config="forgelm_v2")
    server.serve(port=8000)

Task-based concurrent generation:
    POST /v1/tasks                    — create a new task (returns task_id)
    POST /v1/tasks/{task_id}/messages — append a user message + generate
    GET  /v1/tasks/{task_id}          — get task status + history
    GET  /v1/tasks                    — list all active tasks
    DELETE /v1/tasks/{task_id}        — delete a task
"""
import argparse
import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from forge.engine.decide import QuestionValidationError
from forge.engine.model_registry import ModelRegistry, VRAMBudgetExceeded
from forge.engine.session_manager import (
    BatchQueue,
    SessionManager,
    TaskBootConfig,
)
from forge.self_play.discovery.qwen_adapter import (
    IM_END,
    IM_START,
    TOOL_CALL_END,
    TOOL_CALL_START,
    qwen_parse_tool_calls,
    qwen_render_messages,
)
from research.paths import FORGE_TOKENIZER_DIR, V2_CHECKPOINT

logger = logging.getLogger(__name__)


def _parse_reasoning(text: str, model: str = "") -> tuple[str, str | None]:
    """Extract thinking/reasoning blocks from model output.

    Supports DeepSeek-R1 (<think>...</think>), QwQ/Qwen3
    (<|thinking_start|>...<|thinking_end|>), and generic
    <reasoning>...</reasoning> tags.

    Returns (content_without_reasoning, reasoning_text_or_None).
    """
    import re
    patterns = [
        (r'<think>(.*?)</think>', re.DOTALL),
        (r'<\|thinking_start\|>(.*?)<\|thinking_end\|>', re.DOTALL),
        (r'<reasoning>(.*?)</reasoning>', re.DOTALL),
    ]
    for pattern, flags in patterns:
        m = re.search(pattern, text, flags)
        if m:
            reasoning = m.group(1).strip()
            content = re.sub(pattern, '', text, flags=flags).strip()
            return content, reasoning
    return text, None


# ── Pydantic models ──────────────────────────────────────────────────────────

class ToolFunction(BaseModel):
    name: str
    description: str | None = None
    parameters: dict | None = None

class ToolDefinition(BaseModel):
    type: Literal["function"] = "function"
    function: ToolFunction

class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[dict] | None = None
    name: str | None = None
    tool_call_id: str | None = None

class ChatCompletionRequest(BaseModel):
    model: str = "forgelm-v2-jamba"
    messages: list[ChatMessage]
    tools: list[ToolDefinition] | None = None
    tool_choice: Any | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 80
    max_tokens: int = 256
    stream: bool = False
    stop: list[str] | None = None
    repetition_penalty: float = 1.05
    min_p: float = 0.0
    dry_multiplier: float = 0.0
    dry_base: float = 1.75
    dry_allowed_length: int = 2
    dry_penalty_last_n: int = 512
    seed: int | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    lora_adapter: str | None = None

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage | None = None
    delta: dict | None = None
    finish_reason: str | None = None

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)

class CompletionRequest(BaseModel):
    model: str = "forgelm-v2-jamba"
    prompt: str
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 80
    max_tokens: int = 256
    stream: bool = False
    stop: list[str] | None = None
    repetition_penalty: float = 1.05
    min_p: float = 0.0
    dry_multiplier: float = 0.0
    dry_base: float = 1.75
    dry_allowed_length: int = 2
    dry_penalty_last_n: int = 512
    seed: int | None = None
    logprobs: int | None = None
    prompt_logprobs: int | None = None

class EmbeddingRequest(BaseModel):
    model: str = "forgelm-v2-jamba"
    input: str | list[str]
    encoding_format: str = "float"

class RerankingRequest(BaseModel):
    model: str = "forgelm-v2-jamba"
    query: str
    documents: list[str]
    top_n: int | None = None

class InfillRequest(BaseModel):
    model: str = "forgelm-v2-jamba"
    prefix: str
    suffix: str
    max_tokens: int = 128
    temperature: float = 0.2
    top_p: float = 0.9

class MessagesRequest(BaseModel):
    model: str = "forgelm-v2-jamba"
    messages: list[dict]
    system: str | None = None
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 80
    stop_sequences: list[str] | None = None
    seed: int | None = None

class SystemOneQuestion(BaseModel):
    """One typed question in a System One request (TypeSafe wire shape)."""
    type: Literal["noul", "choice", "score"]
    instructions: Any | None = None
    criteria: Any | None = None

class SystemOneRequest(BaseModel):
    """TypeSafe /v1/systemone request: state + typed questions."""
    state: Any
    model: str = ""
    questions: dict[str, SystemOneQuestion]

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "forgeai"

class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]

class RouterRegisterRequest(BaseModel):
    model_id: str
    checkpoint: str
    config_name: str = "forgelm_v2"
    tokenizer_path: str | None = None
    vram_budget_gb: float | None = None

class KVCacheTypeRequest(BaseModel):
    model: str = "forgelm-v2-jamba"
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"

class SleepRequest(BaseModel):
    level: int = Field(default=1, ge=1, le=2)

class LoRALoadRequest(BaseModel):
    model: str = "forgelm-v2-jamba"
    adapter_path: str
    rank: int = 32
    alpha: int | None = None


# ── Task API models ──────────────────────────────────────────────────────────

class CreateTaskRequest(BaseModel):
    model: str = "forgelm-v2-jamba"
    system_prompt: str = ""
    seed: int | None = None  # default seed for this task's generations
    boot_params: dict | None = None  # per-task engine boot configuration
    metadata: dict | None = None

class TaskMessageRequest(BaseModel):
    content: str
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 80
    repetition_penalty: float = 1.05
    seed: int | None = None  # override task's default seed for this message
    stream: bool = False
    stop: list[str] | None = None
    tools: list[ToolDefinition] | None = None

class UpdateBootConfigRequest(BaseModel):
    """Update a task's boot-time engine configuration."""
    kv_cache: str | None = None  # standard, paged, hadamard_int4, snapkv, snapkv_4bit
    decoding: str | None = None  # standard, speculative, batched, mtp_selfspec
    quantize: str | None = None  # None, int8, int4, fp8
    acceleration: str | None = None  # None, cuda_graph, airllm_streaming
    kv_cache_tokens: int | None = None  # Limit KV cache allocation
    kv_bits: int | None = None  # 4 or 8
    mrl_keep_ratio: float | None = None
    use_v0_warm: bool | None = None
    use_progressive_kv: bool | None = None
    use_compile: bool | None = None
    use_triton_conv: bool | None = None
    use_prefix_cache: bool | None = None
    use_spec_attn: bool | None = None
    warmup: bool | None = None

class TaskResponse(BaseModel):
    task_id: str
    model: str
    status: str = "active"
    created_at: float
    messages: int
    total_tokens: int

class TaskListResponse(BaseModel):
    object: str = "list"
    data: list[dict]

class TaskMessageResponse(BaseModel):
    task_id: str
    role: str = "assistant"
    content: str
    created_at: float


# ── Hot-swap API models ──────────────────────────────────────────────────────

class UpdateEngineSettingsRequest(BaseModel):
    """Hot-edit any engine setting. Only set fields are updated."""
    model: str = "forgelm-v2-jamba"
    kv_cache: str | None = None
    decoding: str | None = None
    quantize: str | None = None
    acceleration: str | None = None
    kv_cache_tokens: int | None = None
    kv_bits: int | None = None
    max_context_tokens: int | None = None
    infinite_context: bool | None = None
    default_temperature: float | None = None
    default_max_tokens: int | None = None
    default_top_p: float | None = None
    default_top_k: int | None = None
    default_repetition_penalty: float | None = None
    use_compile: bool | None = None
    use_triton_conv: bool | None = None
    use_prefix_cache: bool | None = None
    use_chunked_prefill: bool | None = None
    use_fused_qk_norm_rope_cache: bool | None = None
    use_seq_split: bool | None = None
    vram_safety_margin_gb: float | None = None
    auto_offload: bool | None = None
    max_batch_size: int | None = None
    batch_timeout_ms: int | None = None

class InfiniteContextRequest(BaseModel):
    model: str = "forgelm-v2-jamba"
    enabled: bool = True
    budget: int = 100_000  # KV cache token budget before eviction

class SwitchStrategyRequest(BaseModel):
    model: str = "forgelm-v2-jamba"
    strategy: str

class ContextLimitRequest(BaseModel):
    model: str = "forgelm-v2-jamba"
    max_tokens: int


class BatchGenerateRequest(BaseModel):
    """Batch generation request for parallel multi-prompt generation."""
    model: str = "forgelm-v2-jamba"
    prompts: list[str]
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 80
    repetition_penalty: float = 1.05


class BatchGenerateResponse(BaseModel):
    model: str
    results: list[str]
    total_tokens: int = 0
    elapsed_ms: float = 0.0


# ── Library API models ───────────────────────────────────────────────────────

class LibrarySaveRequest(BaseModel):
    model: str = "forgelm-v2-jamba"
    content: str
    category: str = "custom"  # failure, win, research, common_data, custom
    tags: list[str] | None = None
    description: str = ""
    triggers: list[str] | None = None
    priority: int = 0
    max_tokens: int = 2048

class LibraryLookupRequest(BaseModel):
    model: str = "forgelm-v2-jamba"
    tags: list[str] | None = None
    category: str | None = None
    limit: int = 50

class LibrarySearchRequest(BaseModel):
    model: str = "forgelm-v2-jamba"
    query: str
    limit: int = 20

class LibraryEntryResponse(BaseModel):
    id: str
    description: str
    category: str
    tags: list[str]
    token_count: int
    priority: int
    access_count: int
    created_at: float
    last_accessed: float
    enabled: bool

class LibraryConfigRequest(BaseModel):
    model: str = "forgelm-v2-jamba"
    enabled: bool | None = None
    injection_budget: int | None = None


class AgentChatRequest(BaseModel):
    """Agentic chat request — model can call built-in tools autonomously."""
    model: str = "forgelm-v2-jamba"
    prompt: str
    max_tokens: int = 512
    max_tool_rounds: int = 5
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 80
    repetition_penalty: float = 1.05
    tools: list[ToolDefinition] | None = None  # extra user-defined tools


class AgentChatResponse(BaseModel):
    model: str
    content: str
    tool_calls: list[dict] = []
    tool_results: list[dict] = []
    rounds: int = 0
    elapsed_ms: float = 0.0


class ExecuteToolRequest(BaseModel):
    """Execute a single built-in tool call directly (no generation)."""
    model: str = "forgelm-v2-jamba"
    name: str
    arguments: dict = {}


class BuiltinToolsResponse(BaseModel):
    """List of available built-in tools."""
    tools: list[dict]


# ── Tool-call helpers ────────────────────────────────────────────────────────

_SPECIAL_MARKERS = (TOOL_CALL_START, TOOL_CALL_END, IM_START, IM_END)


def _parse_tool_calls_openai(raw_text: str) -> tuple[list[dict], str]:
    """Parse tool calls from raw model output. Returns (openai_tool_calls, content)."""
    calls, musing = qwen_parse_tool_calls(raw_text)
    if not calls:
        return [], raw_text.strip()
    openai_calls = []
    for call in calls:
        openai_calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": call["name"],
                "arguments": json.dumps(call.get("arguments", call.get("args", {}))),
            },
        })
    return openai_calls, musing


def _strip_markers(text: str) -> str:
    """Remove special-token markers from text."""
    for marker in _SPECIAL_MARKERS:
        text = text.replace(marker, "")
    return text


def _token_count(engine, text: str) -> int:
    """Count tokens via the engine's tokenizer (usage reporting)."""
    tok = getattr(engine, "tokenizer", None)
    if tok is None or not text:
        return 0
    try:
        if hasattr(tok, "encode"):
            return len(tok.encode(text))
        return int(tok(text, return_tensors="pt").input_ids.shape[-1])
    except Exception:
        return 0


def _usage_dict(engine, prompt: str, completion: str) -> dict:
    """Build an OpenAI-style usage object with real token counts."""
    p = _token_count(engine, prompt)
    c = _token_count(engine, completion)
    return {"prompt_tokens": p, "completion_tokens": c,
            "total_tokens": p + c}


class _ToolCallStreamFilter:
    """Incremental filter that hides tool-call regions from a token stream.

    Feed decoded chunks; returns the text visible to the client (content
    outside ``<tool_call>…</tool_call>``).  Markers split across chunk
    boundaries are handled by buffering a possible partial-marker tail —
    O(1) state, no rescanning of the accumulated string, and text after
    an end marker in the same chunk is preserved (not dropped).
    """

    def __init__(self):
        self.in_tool_call = False
        self._buf = ""
        self.tool_calls_seen = 0

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        out = []
        while self._buf:
            marker = TOOL_CALL_END if self.in_tool_call else TOOL_CALL_START
            if self._buf.startswith(marker):
                self.in_tool_call = not self.in_tool_call
                if self.in_tool_call:
                    self.tool_calls_seen += 1
                self._buf = self._buf[len(marker):]
                continue
            idx = self._buf.find(marker[0])
            if idx < 0:
                if not self.in_tool_call:
                    out.append(self._buf)
                self._buf = ""
                break
            if not self.in_tool_call:
                out.append(self._buf[:idx])
            self._buf = self._buf[idx:]
            # Wait if the remaining buffer could be a split marker.
            if len(self._buf) < len(marker) and marker.startswith(self._buf):
                break
            if self._buf.startswith(marker):
                continue
            # Starts with marker[0] but isn't the marker — consume one char.
            if not self.in_tool_call:
                out.append(self._buf[0])
            self._buf = self._buf[1:]
        return "".join(out)

    def flush(self) -> str:
        """End of stream: emit any buffered tail that wasn't a marker."""
        tail = "" if self.in_tool_call else self._buf
        self._buf = ""
        return tail


# ── Server ───────────────────────────────────────────────────────────────────

class ForgeServer:
    """FastAPI server wrapping ModelRegistry with tool-calling support.

    Features:
      - OpenAI-compatible /v1/chat/completions and /v1/completions
      - Task-based concurrent generation via /v1/tasks endpoints
      - Request batching via BatchQueue (multiple requests → single forward pass)
      - Session management with LRU eviction
      - SSE streaming for both chat and task endpoints
      - Model sleep/wake for VRAM management
    """

    def __init__(self, registry: ModelRegistry | None = None,
                 batch_window_ms: int = 50, max_batch_size: int = 8,
                 max_sessions: int = 64) -> None:
        # ── GPU device selection + contention guard ───────────────────────
        self._gpu_lock_file = None
        self._setup_gpu()

        self.registry = registry or ModelRegistry()
        self.session_manager = SessionManager(max_sessions=max_sessions)
        self.batch_queue = BatchQueue(
            self.registry, self.session_manager,
            batch_window_ms=batch_window_ms,
            max_batch_size=max_batch_size,
            continuous_batching=True,
        )
        self.app = FastAPI(title="ForgeAI Inference Server", version="3.0.0")
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self._setup_routes()

    def _setup_gpu(self):
        """Set CUDA_VISIBLE_DEVICES and detect GPU contention from other servers.

        - If CUDA_VISIBLE_DEVICES is not already set, defaults to "0".
        - Logs which GPU is being used.
        - Creates a temp file-lock per GPU to warn if another forge_server
          is already running on the same device.
        """
        gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES")
        if gpu_id is None:
            gpu_id = "0"
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

        # Log GPU info (best-effort — torch may not be imported yet)
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                logger.info("Using GPU %s: %s", gpu_id, gpu_name)
            else:
                logger.info("CUDA_VISIBLE_DEVICES=%s (CUDA not available)", gpu_id)
        except Exception:
            logger.info("CUDA_VISIBLE_DEVICES=%s", gpu_id)

        # GPU contention guard: file-lock per GPU
        try:
            lock_dir = os.path.join(tempfile.gettempdir(), "forge_gpu_locks")
            os.makedirs(lock_dir, exist_ok=True)
            lock_path = os.path.join(lock_dir, f"gpu_{gpu_id}.lock")
            # Try to create the file exclusively — if it exists, another
            # forge_server may be running on the same GPU.
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                self._gpu_lock_file = lock_path
            except FileExistsError:
                # Check if the PID in the lock file is still alive
                try:
                    with open(lock_path) as f:
                        old_pid = int(f.read().strip())
                    # Cross-platform alive check: os.kill(pid, 0) on Unix,
                    # or ctypes OpenProcess on Windows
                    is_alive = False
                    if os.name == "nt":
                        try:
                            import ctypes
                            kernel32 = ctypes.windll.kernel32
                            SYNCHRONIZE = 0x00100000
                            handle = kernel32.OpenProcess(SYNCHRONIZE, False, old_pid)
                            if handle:
                                kernel32.CloseHandle(handle)
                                is_alive = True
                        except Exception:
                            is_alive = False
                    else:
                        try:
                            os.kill(old_pid, 0)
                            is_alive = True
                        except OSError:
                            is_alive = False
                    if is_alive:
                        logger.warning(
                            "Another forge_server (PID %s) may already be "
                            "running on GPU %s. This may cause VRAM "
                            "contention.", old_pid, gpu_id)
                    else:
                        # Stale lock — take it over
                        fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY)
                        os.write(fd, str(os.getpid()).encode())
                        os.close(fd)
                        self._gpu_lock_file = lock_path
                except (ValueError, OSError):
                    pass  # Can't read lock file — just warn
        except Exception:
            pass  # Non-fatal: lock setup is best-effort

    def _cleanup_gpu_lock(self):
        """Remove the GPU lock file on shutdown."""
        if self._gpu_lock_file is not None:
            try:
                os.unlink(self._gpu_lock_file)
            except OSError:
                pass
            self._gpu_lock_file = None

    def register(self, model_id: str, checkpoint: str, config_name: str,
                 tokenizer_path: str | None = None,
                 vram_budget_gb: float = 0, **kwargs) -> None:
        """Register a model with the registry."""
        tok_path = tokenizer_path or str(FORGE_TOKENIZER_DIR)
        self.registry.register(
            model_id, checkpoint, config_name,
            tokenizer_path=tok_path, vram_budget_gb=vram_budget_gb, **kwargs)

    def serve(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start the HTTP server (blocking)."""
        self.batch_queue.start()
        logger.info("=" * 60)
        logger.info("ForgeAI Inference Server v3.0.0")
        logger.info("Listening on http://%s:%s", host, port)
        logger.info("Models: %s", [m['id'] for m in self.registry.list_models()])
        logger.info("Batch queue: window=%dms, max_batch=%s",
                    self.batch_queue.batch_window * 1000,
                    self.batch_queue.max_batch_size)
        logger.info("Session manager: max=%s tasks",
                    self.session_manager.max_sessions)
        logger.info("Endpoints: /v1/chat/completions, /v1/completions, "
                    "/v1/models, /v1/tasks, /v1/tasks/{id}/messages, "
                    "/v1/tasks/{id}/config (PATCH), /health")
        logger.info("=" * 60)
        try:
            uvicorn.run(self.app, host=host, port=port, log_level="warning")
        finally:
            self.batch_queue.stop()
            self._cleanup_gpu_lock()

    def _build_prompt(self, req: ChatCompletionRequest) -> str:
        """Build prompt from chat messages, using qwen_adapter when tools are present."""
        messages = [m.model_dump() for m in req.messages]
        if req.tools:
            tools = [
                {
                    "name": t.function.name,
                    "description": t.function.description or "",
                    "parameters": t.function.parameters or {},
                }
                for t in req.tools
            ]
            return qwen_render_messages(messages, tools=tools, add_generation_prompt=True)
        # No tools: simple chat template
        return qwen_render_messages(messages, add_generation_prompt=True)

    def _build_anthropic_prompt(self, req) -> str:
        """Build a text prompt from Anthropic-style messages."""
        parts = []
        if req.system:
            parts.append(f"System: {req.system}")
        for msg in req.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(b.get("text", "") for b in content if b.get("type") == "text")
            if role == "user":
                parts.append(f"Human: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant: ")
        return "\n\n".join(parts)

    def _setup_routes(self):
        app = self.app
        registry = self.registry

        @app.get("/health")
        async def health():
            """Health check with Prometheus-compatible metrics."""
            stats = registry.stats()
            return {
                "status": "ok",
                "version": "3.0.0",
                "models_loaded": stats["total_models"],
                "models_awake": stats["awake"],
                "vram_total_gb": round(stats["total_vram_gb"], 2),
                "vram_free_gb": round(stats["free_vram_gb"], 2),
                "active_tasks": len(self.session_manager.list_tasks()),
                "batch_queue_pending": len(self.batch_queue._pending),
            }

        @app.get("/v1/models")
        async def list_models(request: Request):
            """OpenAI-compatible model list.

            TypeSafe SDK clients (identified by the ``X-TypeSafe-SDK``
            header) get the ``{"models": [...]}`` shape their SDK decodes.
            """
            models = registry.list_models()
            if request.headers.get("x-typesafe-sdk"):
                return {
                    "models": [
                        {"name": m["id"],
                         "description": f"{m['config']} (ForgeAI System One)",
                         "release_date": ""}
                        for m in models
                    ]
                }
            now = int(time.time())
            return ModelListResponse(
                data=[
                    ModelInfo(id=m["id"], created=now, owned_by="forgeai")
                    for m in models
                ],
            )

        @app.post("/v1/systemone")
        async def system_one(req: SystemOneRequest, response: Response):
            """TypeSafe System One-compatible typed decisions.

            Accepts the exact wire contract (``state`` + typed ``questions``)
            so ``typesafe_sdk`` clients can point ``base_url`` at this server.
            Model resolution is lenient: an unregistered ``model`` value
            (e.g. ``jev-latest``) falls back to the first registered model.
            """
            models = registry.list_models()
            if not models:
                raise HTTPException(503, "no models registered")
            ids = [m["id"] for m in models]
            model_id = req.model if req.model in ids else ids[0]
            engine = registry.get_engine(model_id)
            if engine is None:
                raise HTTPException(503, f"model '{model_id}' unavailable")
            questions = {k: q.model_dump() for k, q in req.questions.items()}
            try:
                result = engine.decide(req.state, questions)
            except QuestionValidationError as e:
                raise HTTPException(
                    422, detail=[{"loc": ["body", "questions"],
                                  "msg": str(e), "type": "value_error"}])
            result["model"] = model_id
            response.headers["x-typesafe-request-id"] = uuid.uuid4().hex
            return result

        @app.post("/v1/chat/completions")
        async def chat_completions(req: ChatCompletionRequest):
            """OpenAI-compatible chat completions with SSE streaming and tool calling."""
            prompt = self._build_prompt(req)

            if req.stream:
                return StreamingResponse(
                    self._stream_chat(req, prompt),
                    media_type="text/event-stream",
                )

            # Per-request LoRA adapter selection
            lora_swapped = False
            prev_lora = None
            if req.lora_adapter:
                engine = registry.get_engine(req.model)
                if engine is not None:
                    prev_lora = engine.lora_info() if engine.has_lora() else None
                    if prev_lora and prev_lora.get("path") != req.lora_adapter:
                        engine.unload_lora()
                        engine.load_lora(req.lora_adapter)
                        lora_swapped = True
                    elif not prev_lora:
                        engine.load_lora(req.lora_adapter)
                        lora_swapped = True

            # Non-streaming
            raw_text = registry.generate(
                req.model, prompt,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
                min_p=req.min_p,
                dry_multiplier=req.dry_multiplier,
                dry_base=req.dry_base,
                dry_allowed_length=req.dry_allowed_length,
                dry_penalty_last_n=req.dry_penalty_last_n,
                stop=req.stop,
                seed=req.seed,
                logprobs=req.top_logprobs if req.logprobs else None,
                return_logprobs=bool(req.logprobs),
            )

            # Restore previous LoRA adapter if we swapped
            if lora_swapped:
                engine = registry.get_engine(req.model)
                if engine is not None:
                    engine.unload_lora()
                    if prev_lora and prev_lora.get("path"):
                        engine.load_lora(prev_lora["path"], rank=prev_lora.get("rank", 32),
                                        alpha=prev_lora.get("alpha", 64))
            # Handle logprobs return (dict with text + logprobs)
            lp_data = None
            if isinstance(raw_text, dict):
                lp_data = raw_text.get("logprobs")
                raw_text = raw_text.get("text", "")
            tool_calls, content = _parse_tool_calls_openai(raw_text)
            # tool_choice enforcement
            if req.tool_choice == "none" and tool_calls:
                tool_calls = []
                content = raw_text
            elif isinstance(req.tool_choice, dict) and req.tool_choice.get("type") == "function":
                # Force specific function
                fn_name = req.tool_choice.get("function", {}).get("name")
                if fn_name:
                    tool_calls = [{"id": f"call_{uuid.uuid4().hex[:8]}",
                                   "type": "function",
                                   "function": {"name": fn_name, "arguments": "{}"}}]
                    content = None
            elif req.tool_choice == "required" and not tool_calls:
                # Must call a tool — pick the first available
                if req.tools:
                    first_tool = req.tools[0]
                    fn_name = first_tool.function.name if hasattr(first_tool, 'function') else first_tool.get('function', {}).get('name')
                    if fn_name:
                        tool_calls = [{"id": f"call_{uuid.uuid4().hex[:8]}",
                                       "type": "function",
                                       "function": {"name": fn_name, "arguments": "{}"}}]
                        content = None
            # Extract reasoning blocks (DeepSeek-R1/QwQ/Qwen3 thinking)
            reasoning_text = None
            if content:
                content, reasoning_text = _parse_reasoning(content, req.model)
            message: dict[str, Any] = {"role": "assistant", "content": content or None}
            if reasoning_text:
                message["reasoning_content"] = reasoning_text
            if tool_calls:
                message["tool_calls"] = tool_calls
            finish = "tool_calls" if tool_calls else "stop"
            resp_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            # Build response dict directly to support reasoning_content (non-standard field)
            resp = {
                "id": resp_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content or None,
                        "tool_calls": tool_calls or None,
                    } | ({"reasoning_content": reasoning_text} if reasoning_text else {}),
                    "finish_reason": finish,
                    "logprobs": lp_data,
                }],
                "usage": _usage_dict(
                    registry.get_engine(req.model), prompt, raw_text or ""),
            }
            return resp

        @app.post("/v1/completions")
        async def completions(req: CompletionRequest):
            """OpenAI-compatible text completions."""
            if req.stream:
                return StreamingResponse(
                    self._stream_completion(req),
                    media_type="text/event-stream",
                )
            raw_text = registry.generate(
                req.model, req.prompt,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
                min_p=req.min_p,
                dry_multiplier=req.dry_multiplier,
                dry_base=req.dry_base,
                dry_allowed_length=req.dry_allowed_length,
                dry_penalty_last_n=req.dry_penalty_last_n,
                stop=req.stop,
                seed=req.seed,
                logprobs=req.logprobs,
                prompt_logprobs=req.prompt_logprobs,
                return_logprobs=bool(req.logprobs or req.prompt_logprobs),
            )
            # Unwrap dict return when logprobs were requested
            lp_data = None
            if isinstance(raw_text, dict):
                lp_data = raw_text.get("logprobs")
                raw_text = raw_text.get("text", "")
            resp_id = f"cmpl-{uuid.uuid4().hex[:12]}"
            choice = {"index": 0, "text": _strip_markers(raw_text),
                      "finish_reason": "stop"}
            if lp_data:
                choice["logprobs"] = lp_data
            else:
                choice["logprobs"] = None
            return {
                "id": resp_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [choice],
                "usage": _usage_dict(
                    registry.get_engine(req.model), req.prompt,
                    raw_text or ""),
            }

        @app.post("/v1/embeddings")
        async def embeddings(req: EmbeddingRequest):
            """OpenAI-compatible text embeddings."""
            engine = registry.get_engine(req.model)
            if engine is None:
                raise HTTPException(404, f"Model '{req.model}' not found")
            inputs = req.input if isinstance(req.input, list) else [req.input]
            embeddings_list = []
            for text in inputs:
                emb = engine.embed(text)
                embeddings_list.append(emb)
            emb_tokens = sum(_token_count(engine, t) for t in inputs)
            return {
                "object": "list",
                "data": [{"object": "embedding", "embedding": emb, "index": i}
                         for i, emb in enumerate(embeddings_list)],
                "model": req.model,
                "usage": {
                    "prompt_tokens": emb_tokens,
                    "total_tokens": emb_tokens,
                },
            }

        @app.post("/v1/reranking")
        async def reranking(req: RerankingRequest):
            """Rerank documents by relevance to a query."""
            engine = registry.get_engine(req.model)
            if engine is None:
                raise HTTPException(404, f"Model '{req.model}' not found")
            scores = []
            for doc in req.documents:
                score = engine.rerank(req.query, doc)
                scores.append(score)
            # Return sorted indices
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            return {
                "object": "list",
                "data": [{"index": i, "relevance_score": scores[i], "document": {"text": req.documents[i]}}
                         for i in ranked],
                "model": req.model,
            }

        @app.post("/v1/infill")
        async def infill(req: InfillRequest):
            """Fill-in-the-middle code completion."""
            prompt = f"<|fim_prefix|>{req.prefix}<|fim_suffix|>{req.suffix}<|fim_middle|>"
            raw_text = registry.generate(
                req.model, prompt,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                stop=["<|fim_middle|>", "\n\n"],
            )
            return {
                "id": f"infill-{uuid.uuid4().hex[:12]}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{"index": 0, "text": _strip_markers(raw_text),
                             "finish_reason": "stop"}],
                "usage": _usage_dict(
                    registry.get_engine(req.model), prompt, raw_text or ""),
            }

        @app.post("/v1/messages")
        async def messages(req: MessagesRequest):
            """Anthropic-compatible /v1/messages endpoint."""
            prompt = self._build_anthropic_prompt(req)
            raw_text = registry.generate(
                req.model, prompt,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                stop=req.stop_sequences,
                seed=req.seed,
            )
            resp_id = f"msg_{uuid.uuid4().hex[:24]}"
            return {
                "id": resp_id,
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": raw_text}],
                "model": req.model,
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": _token_count(
                        registry.get_engine(req.model), prompt),
                    "output_tokens": _token_count(
                        registry.get_engine(req.model), raw_text or ""),
                },
            }

        @app.post("/v1/batch/completions")
        async def batch_completions(req: BatchGenerateRequest):
            """Batch generation — multiple prompts in one forward pass.

            3-5x faster than serial generation for 2-8 prompts.
            Falls back to serial if batch OOMs.
            """
            engine = registry.get_engine(req.model)
            if engine is None:
                raise HTTPException(404, f"Model '{req.model}' not found")

            import time as _time
            _t0 = _time.perf_counter()
            results = engine.generate_batch(
                req.prompts,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
            )
            _elapsed_ms = (_time.perf_counter() - _t0) * 1000
            _total = sum(len(r.split()) for r in results)

            return BatchGenerateResponse(
                model=req.model,
                results=results,
                total_tokens=_total,
                elapsed_ms=_elapsed_ms,
            )

        # ── Agentic tool endpoints ──────────────────────────────────────

        @app.get("/v1/tools")
        async def list_builtin_tools(model_id: str = "forgelm-v2-jamba"):
            """List all available built-in tools the model can call.

            These tools give the LLM direct access to Library, hot-swap,
            batch generation, and engine introspection features.
            """
            engine = registry.get_engine(model_id)
            if engine is None:
                raise HTTPException(404, f"Model '{model_id}' not found")
            return BuiltinToolsResponse(tools=engine.tools.get_tool_defs())

        @app.post("/v1/tools/execute")
        async def execute_tool(req: ExecuteToolRequest):
            """Execute a single built-in tool call directly (no generation).

            Useful for testing tools or when the client manages the
            conversation loop but wants server-side tool execution.
            """
            engine = registry.get_engine(req.model)
            if engine is None:
                raise HTTPException(404, f"Model '{req.model}' not found")
            result = engine.tools.execute(req.name, req.arguments)
            return {"status": result["status"], "result": result.get("result"),
                    "error": result.get("error")}

        @app.post("/v1/chat/agent")
        async def agent_chat(req: AgentChatRequest):
            """Agentic chat — model autonomously calls built-in tools.

            The model generates a response. If it makes tool calls, they
            are executed server-side and results are fed back. This
            continues for up to `max_tool_rounds` rounds.

            Built-in tools: library_save/search/lookup/get/delete/stats/
            optimize/set_config, engine_set_kv_cache/decoding/context_limit/
            infinite_context/generation_params/feature/apply_changes,
            engine_get_settings/stats/pending, engine_batch_generate,
            engine_generate_adaptive.

            Extra user-defined tools can be passed in `tools` — their
            definitions are included but execution must be handled by
            the client (the model will call them but the server can't
            execute unknown tools).
            """
            engine = registry.get_engine(req.model)
            if engine is None:
                raise HTTPException(404, f"Model '{req.model}' not found")

            extra_tools = None
            if req.tools:
                extra_tools = [
                    {"name": t.function.name,
                     "description": t.function.description or "",
                     "parameters": t.function.parameters or {}}
                    for t in req.tools
                ]

            import time as _time
            _t0 = _time.perf_counter()
            result = engine.generate_with_tools(
                req.prompt,
                max_new_tokens=req.max_tokens,
                max_tool_rounds=req.max_tool_rounds,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
                extra_tools=extra_tools,
            )
            _elapsed = (_time.perf_counter() - _t0) * 1000

            return AgentChatResponse(
                model=req.model,
                content=result["content"],
                tool_calls=result["tool_calls"],
                tool_results=result["tool_results"],
                rounds=result["rounds"],
                elapsed_ms=_elapsed,
            )

        # ── Security management endpoints ───────────────────────────────

        @app.get("/v1/security/config")
        async def get_security_config(model_id: str = "forgelm-v2-jamba"):
            """Get the current security configuration.

            Returns sandbox access rules, file blacklist, website
            whitelist/blacklist, auto mode, and pending request count.
            """
            engine = registry.get_engine(model_id)
            if engine is None:
                raise HTTPException(404, f"Model '{model_id}' not found")
            return engine.tools.security.get_config()

        @app.patch("/v1/security/config")
        async def update_security_config(
            model_id: str = "forgelm-v2-jamba",
            auto_mode: str | None = None,
            set_access_rules: dict[str, str] | None = None,
            remove_access_rules: list[str] | None = None,
            add_file_blacklist: list[str] | None = None,
            remove_file_blacklist: list[str] | None = None,
            add_website_whitelist: list[str] | None = None,
            remove_website_whitelist: list[str] | None = None,
            add_website_blacklist: list[str] | None = None,
            remove_website_blacklist: list[str] | None = None,
        ):
            """Update security configuration.

            All parameters are optional — only provided fields are updated.
            Changes are persisted to sandbox.json.

            auto_mode: "allow" (auto-approve risky), "deny" (auto-deny), "ask" (flag for user)
            set_access_rules: {path: "read_write"|"read_only"|"denied"} — set/update rules
            remove_access_rules: [paths] — remove rules (path defaults to read-only)
            """
            engine = registry.get_engine(model_id)
            if engine is None:
                raise HTTPException(404, f"Model '{model_id}' not found")
            sec = engine.tools.security

            if auto_mode is not None:
                sec.set_auto_mode(auto_mode)
            for path, level in (set_access_rules or {}).items():
                sec.set_access_rule(path, level)
            for path in (remove_access_rules or []):
                sec.remove_access_rule(path)
            for p in (add_file_blacklist or []):
                sec.add_to_file_blacklist(p)
            for p in (remove_file_blacklist or []):
                sec.remove_from_file_blacklist(p)
            for d in (add_website_whitelist or []):
                sec.add_to_website_whitelist(d)
            for d in (remove_website_whitelist or []):
                sec.remove_from_website_whitelist(d)
            for d in (add_website_blacklist or []):
                sec.add_to_website_blacklist(d)
            for d in (remove_website_blacklist or []):
                sec.remove_from_website_blacklist(d)

            return {"status": "ok", "config": sec.get_config()}

        @app.post("/v1/security/reload")
        async def reload_sandbox(model_id: str = "forgelm-v2-jamba"):
            """Reload sandbox config from disk (picks up external edits to sandbox.json)."""
            engine = registry.get_engine(model_id)
            if engine is None:
                raise HTTPException(404, f"Model '{model_id}' not found")
            engine.tools.security.reload_sandbox()
            return {"status": "reloaded", "config": engine.tools.security.get_config()}

        @app.get("/v1/security/access/{path:path}")
        async def check_access(path: str, model_id: str = "forgelm-v2-jamba"):
            """Check the access level for a specific path.

            Returns the access level (read_write, read_only, denied) and
            whether read/write/delete would be allowed.
            """
            engine = registry.get_engine(model_id)
            if engine is None:
                raise HTTPException(404, f"Model '{model_id}' not found")
            sec = engine.tools.security
            level = sec.get_access_level(path)
            read_d = sec.check_file_read(path)
            write_d = sec.check_file_write(path)
            delete_d = sec.check_file_delete(path)
            return {
                "path": path,
                "access_level": level,
                "can_read": read_d.allowed,
                "can_write": write_d.allowed or write_d.needs_permission,
                "can_delete": delete_d.allowed or delete_d.needs_permission,
                "write_reason": write_d.reason if not write_d.allowed else "",
                "delete_reason": delete_d.reason if not delete_d.allowed else "",
            }

        @app.get("/v1/security/pending")
        async def get_pending_permissions(model_id: str = "forgelm-v2-jamba"):
            """Get pending permission requests that need user approval."""
            engine = registry.get_engine(model_id)
            if engine is None:
                raise HTTPException(404, f"Model '{model_id}' not found")
            return {"pending": engine.tools.security.get_pending_requests(),
                    "count": len(engine.tools.security.pending_requests)}

        @app.post("/v1/security/pending/{request_id}/approve")
        async def approve_permission(request_id: str, model_id: str = "forgelm-v2-jamba"):
            """Approve a pending permission request."""
            engine = registry.get_engine(model_id)
            if engine is None:
                raise HTTPException(404, f"Model '{model_id}' not found")
            approved = engine.tools.security.approve_request(request_id)
            if not approved:
                raise HTTPException(404, f"Request '{request_id}' not found")
            return {"status": "approved", "request_id": request_id}

        @app.post("/v1/security/pending/{request_id}/deny")
        async def deny_permission(request_id: str, model_id: str = "forgelm-v2-jamba"):
            """Deny a pending permission request."""
            engine = registry.get_engine(model_id)
            if engine is None:
                raise HTTPException(404, f"Model '{model_id}' not found")
            denied = engine.tools.security.deny_request(request_id)
            if not denied:
                raise HTTPException(404, f"Request '{request_id}' not found")
            return {"status": "denied", "request_id": request_id}

        @app.post("/v1/security/scan")
        async def scan_script(
            model_id: str = "forgelm-v2-jamba",
            content: str = "",
        ):
            """Scan a Python script for dangerous content.

            Returns verdict (allow/needs_permission/refuse) and detailed
            findings including dangerous imports, risky patterns, and
            forbidden patterns.
            """
            engine = registry.get_engine(model_id)
            if engine is None:
                raise HTTPException(404, f"Model '{model_id}' not found")
            return engine.tools.security.scan_script(content)

        @app.post("/v1/models/{model_id}/sleep")
        async def sleep_model(model_id: str, req: SleepRequest = SleepRequest()):
            """Put a model to sleep to free VRAM."""
            if model_id not in [m["id"] for m in registry.list_models()]:
                raise HTTPException(404, f"Model '{model_id}' not found")
            registry.sleep(model_id, level=req.level)
            return {"status": "ok", "model": model_id, "level": req.level}

        @app.post("/v1/lora/load")
        async def lora_load(req: LoRALoadRequest):
            """Load a LoRA adapter onto a model."""
            engine = registry.get_engine(req.model)
            if engine is None:
                raise HTTPException(404, f"Model '{req.model}' not found")
            try:
                n = engine.load_lora(req.adapter_path, rank=req.rank, alpha=req.alpha)
                return {"status": "ok", "model": req.model, "n_adapters": n}
            except Exception as e:
                raise HTTPException(400, str(e)) from e

        @app.post("/v1/lora/unload")
        async def lora_unload(model_id: str = "forgelm-v2-jamba"):
            """Unload LoRA adapter from a model."""
            engine = registry.get_engine(model_id)
            if engine is None:
                raise HTTPException(404, f"Model '{model_id}' not found")
            engine.unload_lora()
            return {"status": "ok", "model": model_id}

        @app.post("/v1/models/{model_id}/wake")
        async def wake_model(model_id: str):
            """Wake a sleeping model."""
            if model_id not in [m["id"] for m in registry.list_models()]:
                raise HTTPException(404, f"Model '{model_id}' not found")
            try:
                registry.wake(model_id)
            except VRAMBudgetExceeded as e:
                raise HTTPException(507, str(e)) from e
            return {"status": "ok", "model": model_id}

        @app.get("/v1/models/{model_id}/stats")
        async def model_stats(model_id: str):
            """Get detailed stats for a specific model."""
            models = registry.list_models()
            for m in models:
                if m["id"] == model_id:
                    return m
            raise HTTPException(404, f"Model '{model_id}' not found")

        # ── Task API: concurrent task-based generation ───────────────────

        @app.post("/v1/tasks")
        async def create_task(req: CreateTaskRequest):
            """Create a new task with its own conversation context.

            Returns a task_id that can be used to send messages and
            maintain conversation continuity across multiple requests.
            Multiple tasks can run concurrently — the batch queue will
            group them into batched forward passes for throughput.
            """
            # Verify model exists
            if req.model not in [m["id"] for m in registry.list_models()]:
                raise HTTPException(404, f"Model '{req.model}' not registered")
            task_id = self.session_manager.create_task(
                model_id=req.model,
                system_prompt=req.system_prompt,
                seed=req.seed,
                boot_config=req.boot_params,
            )
            return {
                "task_id": task_id,
                "model": req.model,
                "status": "active",
                "seed": req.seed,
                "boot_params": req.boot_params or {},
                "created_at": time.time(),
            }

        @app.post("/v1/tasks/{task_id}/messages")
        async def send_task_message(task_id: str, req: TaskMessageRequest):
            """Send a message to a task and get a response.

            The task's conversation history is maintained server-side,
            so each continuation request only needs the new user message.
            Multiple tasks can send messages concurrently — they will be
            batched into a single forward pass when possible.
            """
            session = self.session_manager.get_task(task_id)
            if session is None:
                raise HTTPException(404, f"Task '{task_id}' not found")

            # Build prompt from session history + new message
            messages = []
            if session.system_prompt:
                messages.append({"role": "system", "content": session.system_prompt})
            messages.extend(session.history)
            messages.append({"role": "user", "content": req.content})

            # Render with tools if present
            if req.tools:
                tools = [
                    {
                        "name": t.function.name,
                        "description": t.function.description or "",
                        "parameters": t.function.parameters or {},
                    }
                    for t in req.tools
                ]
                prompt = qwen_render_messages(
                    messages, tools=tools, add_generation_prompt=True)
            else:
                prompt = qwen_render_messages(
                    messages, add_generation_prompt=True)

            # Record user message in session
            self.session_manager.append_message(task_id, "user", req.content)

            # Resolve seed: message-level seed overrides task-level seed
            effective_seed = req.seed if req.seed is not None else session.seed

            if req.stream:
                # Streaming via SSE
                stream_queue = asyncio.Queue()
                future = self.batch_queue.submit(
                    task_id=task_id,
                    model_id=session.model_id,
                    prompt=prompt,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    repetition_penalty=req.repetition_penalty,
                    seed=effective_seed,
                    stop=req.stop,
                    stream=True,
                    stream_queue=stream_queue,
                )
                return StreamingResponse(
                    self._stream_task(task_id, stream_queue, future),
                    media_type="text/event-stream",
                )

            # Non-streaming: submit to batch queue and wait
            future = self.batch_queue.submit(
                task_id=task_id,
                model_id=session.model_id,
                prompt=prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
                seed=effective_seed,
                stop=req.stop,
                stream=False,
            )
            try:
                # Wait for result (with timeout)
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, future.result),
                    timeout=120.0,
                )
            except TimeoutError:
                raise HTTPException(504, "Generation timed out") from None
            except Exception as e:
                raise HTTPException(500, str(e)) from e

            # Parse tool calls if present
            tool_calls, content = _parse_tool_calls_openai(result)

            return {
                "task_id": task_id,
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls or None,
                "created_at": time.time(),
            }

        @app.get("/v1/tasks/{task_id}")
        async def get_task(task_id: str):
            """Get task status and conversation history."""
            session = self.session_manager.get_task(task_id)
            if session is None:
                raise HTTPException(404, f"Task '{task_id}' not found")
            return {
                "task_id": task_id,
                "model": session.model_id,
                "status": "active",
                "created_at": session.created_at,
                "last_used": session.last_used,
                "messages": session.history,
                "total_tokens": session.total_tokens,
                "seed": session.seed,
                "boot_config": session.boot_config.to_dict(),
            }

        @app.patch("/v1/tasks/{task_id}/config")
        async def update_task_config(task_id: str, req: UpdateBootConfigRequest):
            """Update a task's boot-time engine configuration.

            The new config will be applied on the next generation request.
            This allows live reconfiguration of KV cache strategy, decoding
            strategy, quantization, acceleration, and other engine params
            without recreating the task or losing conversation history.

            Only fields that are set (non-None) will be updated; others
            retain their current values.
            """
            session = self.session_manager.get_task(task_id)
            if session is None:
                raise HTTPException(404, f"Task '{task_id}' not found")

            # Merge: only update fields that are set in the request
            current = session.boot_config.to_dict()
            updates = {k: v for k, v in req.model_dump().items()
                       if v is not None}
            current.update(updates)
            new_config = TaskBootConfig.from_dict(current)

            self.session_manager.update_boot_config(task_id, new_config)
            return {
                "status": "updated",
                "task_id": task_id,
                "boot_config": new_config.to_dict(),
            }

        @app.get("/v1/tasks")
        async def list_tasks():
            """List all active tasks."""
            tasks = self.session_manager.list_tasks()
            return {"object": "list", "data": tasks}

        @app.delete("/v1/tasks/{task_id}")
        async def delete_task(task_id: str):
            """Delete a task and free its session context."""
            deleted = self.session_manager.delete_task(task_id)
            if not deleted:
                raise HTTPException(404, f"Task '{task_id}' not found")
            return {"status": "deleted", "task_id": task_id}

        # ── Hot-swap engine config endpoints ──────────────────────────────

        @app.get("/v1/engine/settings")
        async def get_engine_settings(model_id: str = "forgelm-v2-jamba"):
            """Get current engine settings (hot-swappable config)."""
            engine = self.registry.get_engine(model_id)
            if engine is None:
                raise HTTPException(404, f"Model '{model_id}' not found")
            return engine.hotswap.get_settings()

        @app.patch("/v1/engine/settings")
        async def update_engine_settings(req: UpdateEngineSettingsRequest):
            """Hot-edit engine settings without restart.

            Changes take effect on the next generation request.
            Only fields that are set (non-None) will be updated.

            Example:
                PATCH /v1/engine/settings
                {"model": "forgelm-v2-jamba", "kv_cache": "kvzip",
                 "decoding": "eagle3", "temperature": 0.7}
            """
            engine = self.registry.get_engine(req.model)
            if engine is None:
                raise HTTPException(404, f"Model '{req.model}' not found")

            updates = {k: v for k, v in req.model_dump().items()
                       if v is not None and k != "model"}
            engine.hotswap.update_from_dict(updates)
            return {
                "status": "pending",
                "model": req.model,
                "pending_changes": engine.hotswap.get_pending_changes(),
            }

        @app.post("/v1/engine/apply")
        async def apply_engine_settings(model_id: str = "forgelm-v2-jamba"):
            """Force-apply pending hot-swap changes immediately.

            Normally changes are applied lazily on the next generate() call.
            This endpoint forces immediate application (useful for testing).
            """
            engine = self.registry.get_engine(model_id)
            if engine is None:
                raise HTTPException(404, f"Model '{model_id}' not found")
            applied = engine.hotswap.apply_pending()
            return {
                "status": "applied" if applied else "no_changes",
                "model": model_id,
                "settings": engine.hotswap.get_settings(),
            }

        @app.post("/v1/engine/infinite-context")
        async def enable_infinite_context(req: InfiniteContextRequest):
            """Enable infinite context mode with KV cache eviction.

            When enabled, the engine uses eviction-based KV caching to
            maintain unbounded context within the VRAM budget.
            """
            engine = self.registry.get_engine(req.model)
            if engine is None:
                raise HTTPException(404, f"Model '{req.model}' not found")
            engine.hotswap.set_infinite_context(
                enabled=req.enabled, budget=req.budget)
            return {
                "status": "pending",
                "model": req.model,
                "infinite_context": req.enabled,
                "budget": req.budget,
                "pending_changes": engine.hotswap.get_pending_changes(),
            }

        @app.post("/v1/engine/kv-cache")
        async def switch_kv_cache(req: SwitchStrategyRequest):
            """Hot-swap KV cache strategy."""
            engine = self.registry.get_engine(req.model)
            if engine is None:
                raise HTTPException(404, f"Model '{req.model}' not found")
            engine.hotswap.set_kv_cache(req.strategy)
            return {
                "status": "pending",
                "model": req.model,
                "kv_cache": req.strategy,
            }

        @app.post("/v1/engine/kv-cache-type")
        async def set_kv_cache_type(req: KVCacheTypeRequest):
            """Set KV cache quantization type (llama.cpp-style --cache-type-k/v).

            Supported types: f32, f16, bf16, q8_0, q4_0, bf16
            """
            engine = self.registry.get_engine(req.model)
            if engine is None:
                raise HTTPException(404, f"Model '{req.model}' not found")
            type_map = {
                "f32": 32, "f16": 16, "bf16": 16, "q8_0": 8, "q4_0": 4,
            }
            k_bits = type_map.get(req.cache_type_k, 16)
            v_bits = type_map.get(req.cache_type_v, 16)
            try:
                engine.activate(kv_bits=k_bits)
                return {"status": "ok", "cache_type_k": req.cache_type_k,
                        "cache_type_v": req.cache_type_v, "k_bits": k_bits, "v_bits": v_bits}
            except Exception as e:
                raise HTTPException(400, str(e)) from e

        @app.post("/v1/engine/decoding")
        async def switch_decoding(req: SwitchStrategyRequest):
            """Hot-swap decoding strategy."""
            engine = self.registry.get_engine(req.model)
            if engine is None:
                raise HTTPException(404, f"Model '{req.model}' not found")
            engine.hotswap.set_decoding(req.strategy)
            return {
                "status": "pending",
                "model": req.model,
                "decoding": req.strategy,
            }

        @app.post("/v1/engine/context-limit")
        async def set_context_limit(req: ContextLimitRequest):
            """Set the maximum context window (in tokens)."""
            engine = self.registry.get_engine(req.model)
            if engine is None:
                raise HTTPException(404, f"Model '{req.model}' not found")
            engine.hotswap.set_context_limit(req.max_tokens)
            return {
                "status": "pending",
                "model": req.model,
                "max_context_tokens": req.max_tokens,
            }

        @app.get("/v1/engine/pending")
        async def get_pending_changes(model_id: str = "forgelm-v2-jamba"):
            """Check if there are pending hot-swap changes not yet applied."""
            engine = self.registry.get_engine(model_id)
            if engine is None:
                raise HTTPException(404, f"Model '{model_id}' not found")
            return {
                "model": model_id,
                "has_pending": engine.hotswap.has_pending(),
                "pending_changes": engine.hotswap.get_pending_changes(),
            }

        # ── Library endpoints ────────────────────────────────────────────

        @app.get("/v1/library/stats")
        async def library_stats(model_id: str = "forgelm-v2-jamba"):
            """Get library statistics."""
            engine = self.registry.get_engine(model_id)
            if engine is None:
                raise HTTPException(404, f"Model '{model_id}' not found")
            return engine.library_stats()

        @app.post("/v1/library/save")
        async def library_save(req: LibrarySaveRequest):
            """Save an entry to the library (model self-write or user).

            Categories: "failure", "win", "research", "common_data", "custom".
            Content is pre-tokenized on save for instant injection later.
            """
            engine = self.registry.get_engine(req.model)
            if engine is None:
                raise HTTPException(404, f"Model '{req.model}' not found")
            entry_id = engine.library_save(
                content=req.content,
                category=req.category,
                tags=req.tags,
                description=req.description,
                triggers=req.triggers,
                priority=req.priority,
            )
            return {"status": "saved", "entry_id": entry_id}

        @app.post("/v1/library/lookup")
        async def library_lookup(req: LibraryLookupRequest):
            """Lookup entries by tags and/or category."""
            engine = self.registry.get_engine(req.model)
            if engine is None:
                raise HTTPException(404, f"Model '{req.model}' not found")
            entries = engine.library_lookup(
                tags=req.tags, category=req.category, limit=req.limit)
            return {"entries": [
                {"id": e.id, "description": e.description,
                 "category": e.category, "tags": e.tags,
                 "token_count": e.token_count, "priority": e.priority,
                 "access_count": e.access_count, "enabled": e.enabled}
                for e in entries
            ]}

        @app.post("/v1/library/search")
        async def library_search(req: LibrarySearchRequest):
            """Full-text search the library."""
            engine = self.registry.get_engine(req.model)
            if engine is None:
                raise HTTPException(404, f"Model '{req.model}' not found")
            entries = engine.library_search(req.query, limit=req.limit)
            return {"results": [
                {"id": e.id, "description": e.description,
                 "category": e.category, "tags": e.tags,
                 "content_preview": e.content[:200],
                 "token_count": e.token_count, "priority": e.priority}
                for e in entries
            ]}

        @app.get("/v1/library/entry/{entry_id}")
        async def library_get_entry(entry_id: str, model_id: str = "forgelm-v2-jamba"):
            """Get a single library entry by ID."""
            engine = self.registry.get_engine(model_id)
            if engine is None:
                raise HTTPException(404, f"Model '{model_id}' not found")
            entry = engine.library.get(entry_id)
            if entry is None:
                raise HTTPException(404, f"Entry '{entry_id}' not found")
            return {
                "id": entry.id, "content": entry.content,
                "description": entry.description, "category": entry.category,
                "tags": entry.tags, "triggers": entry.triggers,
                "priority": entry.priority, "token_count": entry.token_count,
                "access_count": entry.access_count, "enabled": entry.enabled,
                "created_at": entry.created_at,
                "last_accessed": entry.last_accessed,
            }

        @app.delete("/v1/library/entry/{entry_id}")
        async def library_delete_entry(entry_id: str, model_id: str = "forgelm-v2-jamba"):
            """Delete a library entry."""
            engine = self.registry.get_engine(model_id)
            if engine is None:
                raise HTTPException(404, f"Model '{model_id}' not found")
            deleted = engine.library.delete(entry_id)
            if not deleted:
                raise HTTPException(404, f"Entry '{entry_id}' not found")
            return {"status": "deleted", "entry_id": entry_id}

        @app.patch("/v1/library/config")
        async def library_config(req: LibraryConfigRequest):
            """Configure library injection (enable/disable, set budget)."""
            engine = self.registry.get_engine(req.model)
            if engine is None:
                raise HTTPException(404, f"Model '{req.model}' not found")
            if req.enabled is not None:
                engine.library_set_enabled(req.enabled)
            if req.injection_budget is not None:
                engine.library_set_budget(req.injection_budget)
            return {
                "status": "updated",
                "model": req.model,
                "enabled": engine._library_enabled,
                "injection_budget": engine._library_injection_budget,
            }

        @app.post("/v1/library/optimize")
        async def library_optimize(model_id: str = "forgelm-v2-jamba"):
            """Run library optimization (merge similar, trim, re-index)."""
            engine = self.registry.get_engine(model_id)
            if engine is None:
                raise HTTPException(404, f"Model '{model_id}' not found")
            result = engine.library_optimize()
            return {"status": "optimized", "result": result}

        @app.get("/v1/library/list")
        async def library_list(
            model_id: str = "forgelm-v2-jamba",
            category: str | None = None,
            tag: str | None = None,
            limit: int = 100,
            offset: int = 0,
        ):
            """List library entries with optional filtering."""
            engine = self.registry.get_engine(model_id)
            if engine is None:
                raise HTTPException(404, f"Model '{model_id}' not found")
            entries = engine.library.list_entries(
                category=category, tag=tag, limit=limit, offset=offset)
            return {"entries": entries, "total": len(entries)}

        # ── Router mode: dynamic model registration ───────────────────────
        @app.post("/v1/router/register")
        async def router_register(req: RouterRegisterRequest):
            """Dynamically register a model at runtime (router mode).

            Allows adding models to a running server without restart.
            The model is loaded on first use (lazy) and evicted LRU-style
            when VRAM is needed.
            """
            try:
                engine = self.registry.register(
                    req.model_id, req.checkpoint,
                    config_name=req.config_name,
                    tokenizer_path=req.tokenizer_path,
                    vram_budget_gb=req.vram_budget_gb or 0,
                )
                return {"status": "ok", "model_id": req.model_id,
                        "loaded": engine is not None}
            except Exception as e:
                raise HTTPException(400, str(e)) from e

        @app.delete("/v1/router/{model_id}")
        async def router_unregister(model_id: str):
            """Remove a model from the router."""
            self.registry.unregister(model_id)
            return {"status": "ok", "model_id": model_id}

    async def _stream_chat(self, req: ChatCompletionRequest, prompt: str):
        """SSE streaming generator for chat completions with tool-call detection."""
        resp_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        def _sse_chunk(delta: dict, finish_reason: str | None = None) -> str:
            chunk_data = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": delta,
                             "finish_reason": finish_reason}],
            }
            return f"data: {json.dumps(chunk_data)}\n\n"

        try:
            # Initial role chunk
            yield _sse_chunk({"role": "assistant", "content": ""})

            accumulated = ""
            tc_filter = _ToolCallStreamFilter()
            for text_chunk in self.registry.generate_stream(
                req.model, prompt,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
                min_p=req.min_p,
                dry_multiplier=req.dry_multiplier,
                dry_base=req.dry_base,
                dry_allowed_length=req.dry_allowed_length,
                dry_penalty_last_n=req.dry_penalty_last_n,
                stop=req.stop,
                seed=req.seed,
            ):
                accumulated += text_chunk
                # Incremental marker state machine — handles markers split
                # across chunks and preserves tail text after an end marker.
                visible = tc_filter.feed(text_chunk)
                clean = _strip_markers(visible)
                if clean:
                    yield _sse_chunk({"content": clean})
            tail = _strip_markers(tc_filter.flush())
            if tail:
                yield _sse_chunk({"content": tail})

            # Parse tool calls from full accumulated text
            tool_calls, _ = _parse_tool_calls_openai(accumulated)
            if tool_calls:
                for i, tc in enumerate(tool_calls):
                    yield _sse_chunk({"tool_calls": [{
                        "index": i, "id": tc["id"], "type": "function",
                        "function": {"name": tc["function"]["name"], "arguments": ""},
                    }]})
                    yield _sse_chunk({"tool_calls": [{
                        "index": i,
                        "function": {"arguments": tc["function"]["arguments"]},
                    }]})
                yield _sse_chunk({}, finish_reason="tool_calls")
            else:
                yield _sse_chunk({}, finish_reason="stop")
            yield "data: [DONE]\n\n"

        except Exception as e:
            error_data = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": {},
                             "finish_reason": "error"}],
                "error": str(e),
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"

    async def _stream_task(self, task_id: str, stream_queue: asyncio.Queue,
                           future):
        """SSE streaming generator for task-based generation."""
        resp_id = f"task-{task_id}"
        created = int(time.time())

        def _sse_chunk(delta: dict, finish_reason: str | None = None) -> str:
            chunk_data = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "task_id": task_id,
                "choices": [{"index": 0, "delta": delta,
                             "finish_reason": finish_reason}],
            }
            return f"data: {json.dumps(chunk_data)}\n\n"

        try:
            yield _sse_chunk({"role": "assistant", "content": ""})

            while True:
                chunk = await stream_queue.get()
                if chunk is None:  # sentinel = end of stream
                    break
                clean = _strip_markers(chunk)
                if clean:
                    yield _sse_chunk({"content": clean})

            yield _sse_chunk({}, finish_reason="stop")
            yield "data: [DONE]\n\n"
        except Exception as e:
            error_data = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "task_id": task_id,
                "choices": [{"index": 0, "delta": {},
                             "finish_reason": "error"}],
                "error": str(e),
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"

    async def _stream_completion(self, req: CompletionRequest):
        """SSE streaming generator for text completions."""
        resp_id = f"cmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        def _sse_chunk(text: str = "", finish_reason: str | None = None) -> str:
            chunk_data = {
                "id": resp_id,
                "object": "text_completion",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "text": text,
                             "finish_reason": finish_reason}],
            }
            return f"data: {json.dumps(chunk_data)}\n\n"

        try:
            for text_chunk in self.registry.generate_stream(
                req.model, req.prompt,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
                min_p=req.min_p,
                dry_multiplier=req.dry_multiplier,
                dry_base=req.dry_base,
                dry_allowed_length=req.dry_allowed_length,
                dry_penalty_last_n=req.dry_penalty_last_n,
                stop=req.stop,
                seed=req.seed,
            ):
                clean = _strip_markers(text_chunk)
                if clean:
                    yield _sse_chunk(text=clean)
            yield _sse_chunk(finish_reason="stop")
            yield "data: [DONE]\n\n"
        except Exception as e:
            error_data = {
                "id": resp_id,
                "object": "text_completion",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "text": "",
                             "finish_reason": "error"}],
                "error": str(e),
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"


# ── CLI ──────────────────────────────────────────────────────────────────────

# Default model registrations
DEFAULT_MODELS = {
    "forgelm-v2-jamba": {
        "checkpoint": str(V2_CHECKPOINT),
        "config": "forgelm_v2",
        "tokenizer": str(FORGE_TOKENIZER_DIR),
        # bf16 checkpoint ~6.1GB + KV cache + activation headroom on 12GB
        "vram_gb": 8.0,
    },
}


def main():
    # Opt-in runtime configuration (import of `forge` is side-effect-free).
    from forge.runtime.configure import configure
    configure()

    parser = argparse.ArgumentParser(description="ForgeAI Inference Server v3.0")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--models", type=str, default="forgelm-v2-jamba",
                        help="Comma-separated model IDs to load")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override checkpoint path")
    parser.add_argument("--config", type=str, default="forgelm_v2",
                        help="Model config preset")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Override tokenizer path")
    parser.add_argument("--vram-gb", type=float, default=0,
                        help="VRAM budget in GB (0=auto)")
    parser.add_argument("--batch-window-ms", type=int, default=50,
                        help="Batch collection window in ms (default 50)")
    parser.add_argument("--max-batch-size", type=int, default=8,
                        help="Maximum requests per batch (default 8)")
    parser.add_argument("--max-sessions", type=int, default=64,
                        help="Maximum concurrent task sessions (default 64)")
    args = parser.parse_args()

    server = ForgeServer(
        batch_window_ms=args.batch_window_ms,
        max_batch_size=args.max_batch_size,
        max_sessions=args.max_sessions,
    )

    # Register models
    model_ids = [m.strip() for m in args.models.split(",")]
    for mid in model_ids:
        if mid in DEFAULT_MODELS and not args.checkpoint:
            spec = DEFAULT_MODELS[mid]
            server.register(
                mid, spec["checkpoint"], spec["config"],
                tokenizer_path=spec.get("tokenizer"),
                vram_budget_gb=args.vram_gb or spec.get("vram_gb", 0),
            )
        elif args.checkpoint:
            server.register(
                mid, args.checkpoint, args.config,
                tokenizer_path=args.tokenizer,
                vram_budget_gb=args.vram_gb,
            )
        else:
            logger.warning("No checkpoint for '%s', skipping. "
                           "Use --checkpoint to specify.", mid)

    server.serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
