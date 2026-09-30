"""ForgeEvolve engine: the phase loop.

  Phase 1: GENERATE  — N generators produce candidates (batched GPU forward)
  Phase 2: FILTER    — surrogate predicts scores, top-K selected (GPU)
  Phase 3: SCORE     — real evaluation of K candidates (domain-specific)
  Phase 4: TRAIN     — update generators + surrogate + archive (GPU)
  Phase 5: REPEAT

CUDA: generators + surrogate run on GPU when available. Domain evaluation
may use GPU independently (e.g., quant domain does FP4 ops on GPU).

Usage:
  from forge.evolution import ForgeEvolve, ForgeEvolveConfig
  from forge.evolution.domains.synthetic import SyntheticDomain

  cfg = ForgeEvolveConfig(
      domain=SyntheticDomain(...),
      n_generators=500,
      filter_ratio=20,
      generations=50,
  )
  engine = ForgeEvolve(cfg)
  results = engine.run()
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

from .archive import MapElitesArchive
from .database import FindingsDB
from .domains import BaseDomain
from .generators import GeneratorConfig, GeneratorPopulation, TemplateGenerator
from .surrogate import SurrogateModel
from .trainer import GeneratorTrainer

# ── CPU worker globals (set by _cpu_worker_init in each worker process) ──
_CPU_DOMAIN = None


def _cpu_worker_init(domain_cls, seq_len: int):
    """Initialize a CPU worker process with its own domain copy.

    Runs once per worker at pool creation. Creates a CPU-only domain
    so the worker can evaluate configs without touching the GPU.
    Handles both heavy domains (seq_len, seed, device args) and
    lightweight domains (no args).
    """
    global _CPU_DOMAIN
    import torch
    torch.set_num_threads(2)
    # Try with full args first, fall back to no-arg constructor
    try:
        _CPU_DOMAIN = domain_cls(seq_len=seq_len, seed=43, device=torch.device("cpu"))
    except TypeError:
        _CPU_DOMAIN = domain_cls()


def _cpu_eval(config: dict) -> dict:
    """Evaluate a config on CPU in a worker process."""
    global _CPU_DOMAIN
    return _CPU_DOMAIN.evaluate(config)


@dataclass
class ForgeEvolveConfig:
    """Configuration for a ForgeEvolve run."""
    # Domain (provides evaluate + config space)
    domain: Any  # BaseDomain

    # Generator population
    n_generators: int = 500
    noise_dim: int = 16
    context_dim: int = 32
    hidden_dim: int = 64
    mutation_rate: float = 0.1

    # Filtering
    filter_ratio: int = 20        # N_generated / N_evaluated
    min_evaluate: int = 10        # minimum candidates to evaluate per gen
    max_evaluate: int = 150       # maximum candidates to evaluate per gen (raised for throughput)
    elite_mutants: int = 0        # per-gen domain-space mutants of archive elites
                                  # (0=off; domains define propose_elite_mutants)

    # Training
    generator_lr: float = 1e-3
    surrogate_lr: float = 1e-3
    surrogate_mode: str = "mlp"   # "mlp" or "gp"

    # Search
    generations: int = 50
    exploration: float = 0.3      # 0=exploit, 1=explore
    time_budget_s: float = 0      # 0 = no limit
    convergence_patience: int = 0  # 0 = disabled; >0 = stop early if best
                                   # score doesn't improve for N consecutive
                                   # generations (evolution convergence detector)
    # Adaptive population: grow generators when plateauing, shrink when stable
    adaptive_population: bool = True  # auto-tune n_generators per domain
    pop_growth_rate: float = 0.2  # grow by 20% on plateau
    pop_shrink_rate: float = 0.1  # shrink by 10% when improving steadily
    pop_min: int = 50             # minimum population size
    pop_max: int = 2000           # maximum population size
    # Dynamic domain spawning: when a domain converges, spawn a refinement
    # domain that narrows the search around the best config. This enables
    # infinite-depth search without the 57-domain limit.
    enable_refinement: bool = True
    refinement_max_depth: int = 5  # max recursive refinement depth
    refinement_narrowing: float = 0.2  # each level narrows to ±20% of parent range
    # Novelty search: alternate between novelty (explore) and quality (exploit)
    # phases to prevent plateaus. Based on Lehman & Stanley novelty search +
    # composite novelty pulsation (GECCO 2020).
    enable_novelty: bool = True
    novelty_k_neighbors: int = 5
    novelty_pulse_len: int = 3   # generations of novelty search per cycle
    quality_pulse_len: int = 5   # generations of quality search per cycle
    novelty_bonus: float = 1.5   # bonus added to surrogate during novelty phase

    # Device
    device: str = "auto"          # "auto", "cuda", "cpu"

    # Database
    db_path: str = "forge_evolve.db"
    run_id: str = ""              # empty = auto-generate
    warm_start: bool = True       # load past findings for this domain

    # Parallelism
    parallel_eval: int = 1        # GPU eval parallelism

    # ── Compute split: foreground gen models + background score checks ──
    # When the domain uses an LLM gen model (gen_model_type="llm" in JSON spec),
    # the engine batches gen-model inference on the GPU (foreground) and runs
    # the SharedCheckerModel scoring asynchronously (background) to overlap
    # compute.  Set enable_compute_split=True to activate.
    enable_compute_split: bool = False
    gen_model_manager: Any = None   # GenModelManager instance (optional)
    checker_model: Any = None       # SharedCheckerModel instance (optional)
    # Background checker uses a ThreadPoolExecutor so the LLM-as-judge calls
    # overlap with the next generation's gen-model inference.
    checker_n_workers: int = 2

    # Logging
    verbose: bool = True
    log_every: int = 1


class ForgeEvolve:
    """The main evolutionary search engine."""

    def __init__(self, cfg: ForgeEvolveConfig):
        self.cfg = cfg
        self.domain = cfg.domain
        self.output_dim = self.domain.output_dim()

        # Enable TF32 tensor cores for faster matmul on Ampere+ (RTX 5070)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Device selection
        if cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif cfg.device == "cuda":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Build generator population (batched, on device)
        gen_cfg = GeneratorConfig(
            noise_dim=cfg.noise_dim,
            context_dim=cfg.context_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=self.output_dim,
            n_generators=cfg.n_generators,
            mutation_rate=cfg.mutation_rate,
        )

        template = None
        discrete_choices = self.domain.discrete_choices()
        if discrete_choices:
            template = TemplateGenerator(discrete_choices)

        self.population = GeneratorPopulation(gen_cfg, template, device=self.device)

        # Surrogate (on device)
        self.surrogate = SurrogateModel(
            input_dim=self.output_dim,
            mode=cfg.surrogate_mode,
            lr=cfg.surrogate_lr,
            device=self.device,
        )

        # Archive
        self.archive = MapElitesArchive(self.domain.behavioral_dims())

        # Domain factory for dynamic refinement spawning
        from forge.evolution.domain_factory import DomainFactory
        self.domain_factory = DomainFactory(
            max_depth=cfg.refinement_max_depth,
            narrowing=cfg.refinement_narrowing,
        )
        self.pending_refinements: list[BaseDomain] = []

        # Novelty search: behavioral diversity to prevent plateaus
        from forge.evolution.novelty_search import NoveltyConfig, NoveltySearch
        novelty_cfg = NoveltyConfig(
            enabled=cfg.enable_novelty,
            k_neighbors=cfg.novelty_k_neighbors,
            pulse_novelty_len=cfg.novelty_pulse_len,
            pulse_quality_len=cfg.quality_pulse_len,
            novelty_bonus=cfg.novelty_bonus,
        )
        self.novelty = NoveltySearch(novelty_cfg)

        # Trainer (uses batched generator, on device)
        self.trainer = GeneratorTrainer(
            self.population.batched_gen, lr=cfg.generator_lr,
            device=self.device,
        )

        # State
        self.generation = 0
        self.all_results: list[dict] = []
        self.max_results: int = 10000  # cap to prevent OOM on long runs
        self.discoveries: list[dict] = []
        self.start_time = 0.0

        # Config dedup cache: prevents re-evaluating identical configs within
        # a run AND prevents duplicate discoveries being saved to DB.
        # Key = canonical config string, value = (score, behavioral, metadata)
        self._eval_cache: dict[str, tuple] = {}

        # CPU/GPU pair processing: separate processes for true parallelism
        # GPU process: generators + surrogate + GPU-side evals
        # CPU processes: CPU-side evals (bypass GIL entirely)
        self.cpu_pool = None
        self.n_cpu_workers = 0
        if cfg.parallel_eval > 1 and self.device.type == "cuda":
            try:
                import multiprocessing as mp
                # Use spawn to avoid CUDA fork issues
                ctx = mp.get_context("spawn")
                self.n_cpu_workers = cfg.parallel_eval - 1
                self.cpu_pool = ctx.Pool(
                    processes=self.n_cpu_workers,
                    initializer=_cpu_worker_init,
                    initargs=(type(self.domain), self.domain.seq_len if hasattr(self.domain, "seq_len") else 2048),
                )
            except Exception as e:
                if cfg.verbose:
                    print(f"[CPU pool] Failed to start: {e}, falling back to GPU-only")
                self.cpu_pool = None
                self.n_cpu_workers = 0

        # Database
        self.db = FindingsDB(cfg.db_path)
        self.run_id = cfg.run_id or f"{self.domain.name()}_{int(time.time())}"
        self._warm_started = False
        self._pending_discoveries = []  # batched DB saves

        # ── Compute split: gen model + checker wiring ──
        self._gen_mgr = cfg.gen_model_manager
        self._checker = cfg.checker_model
        self._checker_pool = None
        if cfg.enable_compute_split:
            import concurrent.futures
            self._checker_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=cfg.checker_n_workers,
                thread_name_prefix="checker",
            )
            # Auto-create defaults if not provided
            if self._gen_mgr is None:
                from .gen_model_manager import GenModelManager
                self._gen_mgr = GenModelManager(db=self.db)
            if self._checker is None:
                from .checker_model import get_checker
                self._checker = get_checker()
            # Wire gen model into domain if it supports set_gen_model
            if hasattr(self.domain, "set_gen_model") and self._gen_mgr is not None:
                gm = getattr(self._gen_mgr, "_model", None)
                if gm is not None:
                    self.domain.set_gen_model(gm)
                    self._log(f"[ComputeSplit] gen model attached to domain "
                              f"({gm.param_count()} params)")
            self._log(f"[ComputeSplit] enabled — "
                      f"checker={'yes' if self._checker else 'no'}, "
                      f"gen_mgr={'yes' if self._gen_mgr else 'no'}")

        # ── Canonical load: ALWAYS load the best-ever generators + surrogate
        # for this domain, regardless of warm_start. This is the permanent
        # knowledge layer — findings from all past runs (any profile, any
        # DB) accumulate here. The search starts from the best-known point.
        # Skip canonical load for Generic domains (heuristic evaluator)
        is_generic = "Generic_" in self.domain.name()
        canonical_best = self.db.get_canonical_best_score(self.domain.name())
        if not is_generic and self.db.load_canonical_generators(self.domain.name(),
                                             self.population.batched_gen):
            self._warm_started = True
            self._log(f"Loaded canonical generators for '{self.domain.name()}'"
                      + (f" (past best={canonical_best:.4f})" if canonical_best else ""))
        if not is_generic and self.db.load_canonical_surrogate(self.domain.name(), self.surrogate):
            if not self._warm_started:
                self._warm_started = True
            self._log(f"Loaded canonical surrogate for '{self.domain.name()}'")

        # Warm-start from past findings (per-run generators + discoveries)
        if cfg.warm_start:
            n_seeded = self.db.seed_from_past(
                self.domain.name(),
                self.population.batched_gen,
                self.surrogate,
            )
            if n_seeded > 0:
                self._warm_started = True
                self._log(f"Warm-started from {n_seeded} past findings "
                          f"(generators + surrogate loaded)")
            else:
                # Cross-domain transfer: if no same-domain history, borrow
                # generators from a different domain with matching output_dim.
                # Skip for Generic domains (heuristic, not real performance)
                if not is_generic:
                    od = self.domain.output_dim()
                    n_xfer = self.db.seed_cross_domain(
                        self.domain.name(),
                        self.population.batched_gen,
                        self.surrogate,
                        od,
                    )
                    if n_xfer > 0:
                        self._warm_started = True
                        self._log(f"Cross-domain transfer from {n_xfer} source domain(s) "
                                  f"(output_dim={od})")
            # Also load past discoveries as archive seeds
            # Cross-run cache: use stored scores from DB instead of re-evaluating
            past = self.db.query_best_configs(self.domain.name(), limit=20)
            n_cached = 0
            n_reevaled = 0
            for p in past:
                key = self._config_key(p["config"])
                if key in self._eval_cache:
                    continue  # already evaluated as a seed
                # Use stored score from DB (cross-run cache) — avoids
                # re-evaluating configs that were already scored in past runs.
                stored_score = p.get("score")
                stored_behavioral = p.get("behavioral", (0,))
                stored_metadata = p.get("metadata", {})
                if stored_score is not None:
                    self._eval_cache[key] = (stored_score,
                                             stored_behavioral,
                                             stored_metadata)
                    self.archive.add(p["config"], stored_score,
                                     stored_behavioral, -1)
                    n_cached += 1
                else:
                    result = self.domain.evaluate(p["config"])
                    self._eval_cache[key] = (result["score"],
                                             result.get("behavioral", (0,)),
                                             result.get("metadata", {}))
                    self.archive.add(p["config"], result["score"],
                                     result.get("behavioral", (0,)), -1)
                    n_reevaled += 1
            if past:
                self._log(f"Loaded {len(past)} past discoveries into archive "
                          f"({n_cached} cached, {n_reevaled} re-evaluated)")

        # Seed from domain's known-good configs
        seed_configs = self.domain.seed_configs()
        if seed_configs:
            self._log(f"Evaluating {len(seed_configs)} seed configs...")
            seed_params = []
            seed_scores = []
            for sc in seed_configs:
                key = self._config_key(sc)
                if key in self._eval_cache:
                    continue  # already evaluated (e.g., from past discoveries)
                result = self.domain.evaluate(sc)
                self._eval_cache[key] = (result["score"],
                                         result.get("behavioral", (0,)),
                                         result.get("metadata", {}))
                self.archive.add(sc, result["score"],
                                 result.get("behavioral", (0,)), -1)
                self.all_results.append({
                    "generation": -1,
                    "config": sc,
                    "score": result["score"],
                    "behavioral": result.get("behavioral", (0,)),
                    "metadata": result.get("metadata", {}),
                })
                # Persist seeds so query_best_configs reflects the archive's
                # best (dedup updates the row if a better score comes later)
                self._pending_discoveries.append({
                    "generation": -1,
                    "config": sc,
                    "score": result["score"],
                    "behavioral": result.get("behavioral", (0,)),
                    "metadata": result.get("metadata", {}),
                })
                # Encode for surrogate training
                params = self.domain.encode(sc)
                if params is not None:
                    seed_params.append(params)
                    seed_scores.append(result["score"])

            # Train surrogate on seeds immediately (not random anymore)
            if seed_params:
                param_tensor = torch.stack(seed_params).to(self.device)
                score_tensor = torch.tensor(seed_scores, dtype=torch.float32)
                try:
                    self.surrogate.train(param_tensor, score_tensor, epochs=5)
                    self._log(f"Surrogate trained on {len(seed_params)} seeds "
                              f"(best seed score={max(seed_scores):.4f})")
                except Exception as e:
                    self._log(f"Surrogate seed training skipped: {e}")

            # Update context with best seed
            if seed_scores:
                best_idx = int(np.argmax(seed_scores))
                self.population.update_context(seed_params[best_idx])

    # ── Config dedup ──────────────────────────────────────────────────

    @staticmethod
    def _config_key(config: dict) -> str:
        """Canonical hashable key for a config dict.
        Sorts keys and normalizes values to handle int/str equivalence
        (e.g., 32 and '32' from JSON round-trip)."""
        normalized = {}
        for k in sorted(config.keys()):
            v = config[k]
            # Normalize numeric strings to their numeric form
            if isinstance(v, str):
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
            # Numpy scalar → python scalar
            if hasattr(v, 'item') and not hasattr(v, '__len__'):
                v = v.item()
            # Round floats to 6 decimals to avoid hash misses from FP noise
            if isinstance(v, float):
                v = round(v, 6)
            # Numpy arrays / lists → tuple of rounded floats
            # (but not strings — they have no numeric values to round)
            if isinstance(v, str):
                pass  # keep as-is
            elif hasattr(v, 'tolist'):
                v_list = v.tolist()
                if isinstance(v_list, (list, tuple)):
                    v = tuple(round(float(x), 6) for x in v_list)
                else:
                    v = round(float(v_list), 6)
            elif isinstance(v, (list, tuple)):
                v = tuple(round(float(x), 6) if isinstance(x, (int, float))
                          else x for x in v)
            normalized[k] = v
        return json.dumps(normalized, sort_keys=True, default=str)

    def _eval_with_dedup(self, config: dict) -> dict | None:
        """Evaluate a config, returning cached result if we've seen it before.
        Returns None if this is a duplicate (caller should skip)."""
        key = self._config_key(config)
        if key in self._eval_cache:
            return None  # duplicate — skip
        result = self._safe_eval(config)
        self._eval_cache[key] = (result["score"],
                                 result.get("behavioral", (0,)),
                                 result.get("metadata", {}))
        return result

    def _gpu_batch_eval(self, configs: list[dict]) -> list[dict]:
        """Evaluate configs on GPU with Triton batched kernels + CUDA stream overlap.

        Priority:
        1. Triton batched eval (if domain supports it) — single GPU launch for all configs
        2. Threaded eval with CUDA streams — overlap kernel launches across configs
        3. Sequential eval — fallback for small batches
        """
        import torch

        if len(configs) <= 2:
            return [self._safe_eval(cfg) for cfg in configs]

        # ── Try Triton batched evaluation first ──
        # This processes all configs in a single GPU operation, eliminating
        # Python/kernel-launch overhead entirely for supported domains.
        if self.device.type == "cuda" and not hasattr(self, '_batched_eval'):
            try:
                from forge.evolution.triton_batched import BatchedEvaluator
                self._batched_eval = BatchedEvaluator(self.domain)
            except Exception:
                self._batched_eval = None

        if (self._batched_eval is not None and self._batched_eval.can_batch()
                and len(configs) >= 4):
            try:
                results = self._batched_eval.batch_evaluate(configs)
                if all(r is not None for r in results):
                    return results
            except Exception:
                logger.debug("Batched eval failed, falling through to threaded eval", exc_info=True)

        # Fixed chunk size: each eval uses ~100MB, reserve 2GB for generators
        # On 12GB GPU: (12GB - 2GB) / (100MB * 1.5) = ~66 concurrent evals
        # Cap at 8 for thread safety
        # VRAM-aware: reduce chunk if VRAM is already heavily used
        if not hasattr(self, '_eval_chunk'):
            if self.device.type == "cuda":
                total = torch.cuda.get_device_properties(0).total_memory
                allocated = torch.cuda.memory_allocated(0)
                available = total - allocated
                # Reserve 1GB headroom for eval workspace
                safe_budget = max(available - 1 * 1024**3, 256 * 1024**2)
                self._eval_chunk = max(1, min(8, int(safe_budget / (150 * 1024**2))))
            else:
                self._eval_chunk = len(configs)

        chunk = self._eval_chunk

        # For small batches, just run sequentially (no thread overhead)
        if len(configs) <= chunk:
            return [self._safe_eval(cfg) for cfg in configs]

        # Chunked evaluation with threaded eval for CUDA stream overlap
        results = []
        for i in range(0, len(configs), chunk):
            batch = configs[i:i + chunk]
            # Use threaded eval within each chunk for kernel overlap
            if self.device.type == "cuda" and len(batch) > 2:
                results.extend(self._threaded_eval(batch))
            else:
                results.extend(self._safe_eval(cfg) for cfg in batch)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return results

    def _safe_eval(self, config: dict) -> dict:
        """Evaluate a single config with OOM recovery."""
        try:
            return self.domain.evaluate(config)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                import torch
                torch.cuda.empty_cache()
                # Return a penalty score so the search doesn't crash
                return {"score": -1e9, "behavioral": (0.0,) * len(self.domain.behavioral_dims()),
                        "metadata": {"oom": True}}
            raise

    def _compute_split_eval(self, configs: list[dict]) -> tuple[list[dict], list[dict]]:
        """Compute-split evaluation for LLM gen-model domains.

        Foreground: gen model generates answers (batched on GPU).
        Background: SharedCheckerModel scores answers asynchronously.

        The domain.evaluate() already calls the gen model internally (via
        set_gen_model).  This method wraps the evaluation so that:
        1. Gen-model inference (foreground) runs in a thread pool to overlap
           CUDA kernel launches.
        2. After each eval completes, if the checker is available and the
           domain's spec has checker_type="llm_judge", the checker score is
           submitted to the background pool and merged when ready.

        Returns (results, configs) in the same order as input.
        """
        # Phase A: foreground gen-model inference (domain.evaluate calls gen model)
        if self.device.type == "cuda" and len(configs) > 2:
            raw_results = self._threaded_eval(configs)
        else:
            raw_results = [self._safe_eval(cfg) for cfg in configs]

        # Phase B: background checker scoring (LLM-as-judge, optional)
        if self._checker is None or self._checker_pool is None:
            return raw_results, configs

        # Submit checker calls to background pool
        futures = []
        for i, (cfg, res) in enumerate(zip(configs, raw_results)):
            meta = res.get("metadata", {})
            problem = meta.get("problem", "")
            model_answer = meta.get("model_answer", "")
            real_answer = meta.get("real_answer")
            if not problem or not model_answer:
                futures.append(None)
                continue
            req_text = (f"Problem: {problem}\n"
                        f"Expected: {real_answer}\n"
                        f"Answer: {model_answer}")
            req = {
                "prompt": req_text,
                "requirements": str(real_answer) if real_answer is not None else "",
                "expected": str(real_answer) if real_answer is not None else "",
            }
            fut = self._checker_pool.submit(self._checker.check, req)
            futures.append(fut)

        # Merge checker scores (non-blocking — if not ready, skip)
        for i, fut in enumerate(futures):
            if fut is None:
                continue
            try:
                cscore = fut.result(timeout=0.1)
                raw_results[i]["metadata"]["checker_score"] = cscore
            except Exception:
                logger.debug("Checker score not ready, heuristic score stands", exc_info=True)

        return raw_results, configs

    def _threaded_eval(self, configs: list[dict]) -> list[dict]:
        """Evaluate configs using threads with CUDA streams for kernel overlap.

        Each thread gets its own CUDA stream so kernels from multiple evals
        can overlap on the GPU, keeping utilization high. The GIL is released
        during CUDA calls, so threads achieve real parallelism.
        """
        from concurrent.futures import ThreadPoolExecutor

        import torch

        # More threads = more kernel overlap. Cap at len(configs) and 8.
        n_threads = min(8, len(configs))
        results = [None] * len(configs)

        def _eval_one(idx_cfg):
            idx, cfg = idx_cfg
            try:
                if self.device.type == "cuda":
                    stream = torch.cuda.Stream()
                    with torch.cuda.stream(stream):
                        result = self.domain.evaluate(cfg)
                    stream.synchronize()
                else:
                    result = self.domain.evaluate(cfg)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    result = {"score": -1e9, "behavioral": (0.0,) * len(self.domain.behavioral_dims()),
                              "metadata": {"oom": True}}
                else:
                    raise
            return idx, result

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            for idx, result in pool.map(_eval_one, enumerate(configs)):
                results[idx] = result

        return results

    def _spawn_refinement(self):
        """Spawn a refinement domain from the converged parent.

        Creates a narrowed search domain centered on the best config,
        registers it in the factory, and queues it for the next loop.
        """
        if not self.archive.best_entry:
            return
        best = self.archive.best_entry
        best_config = best.config
        best_params = self.domain.encode(best_config)

        # Determine refinement depth from parent name
        parent_name = self.domain.name()
        depth = 1
        if "_refine_d" in parent_name:
            depth = int(parent_name.split("_refine_d")[-1]) + 1

        child = self.domain_factory.spawn_refinement(
            parent=self.domain,
            best_config=best_config,
            best_params=best_params,
            depth=depth,
        )
        if child is not None:
            self.domain_factory.register(child)
            self.pending_refinements.append(child)
            self._log(
                f"  [Refinement] Spawned {child.name()} "
                f"(depth={depth}, narrowing=±{child.narrowing:.1%})")

    def _log(self, msg: str):
        if self.cfg.verbose:
            elapsed = time.time() - self.start_time if self.start_time else 0
            print(f"[Gen {self.generation:3d} | {elapsed:6.1f}s] {msg}")

    def run(self) -> dict:
        """Run the full evolutionary search loop."""
        self.start_time = time.time()
        self._prev_best_score = float('-inf')
        self._convergence_counter = 0
        self._pop_stable_count = 0
        self._pop_plateau_count = 0
        self._log(f"Starting ForgeEvolve: {self.cfg.n_generators} generators, "
                  f"filter_ratio={self.cfg.filter_ratio}, "
                  f"domain={self.domain.name()}, device={self.device}"
                  + (f", CPU_pool={self.n_cpu_workers} workers" if self.cpu_pool else ""))

        # ── Pre-loop: if unapplied discoveries exist in DB, train on them first ──
        # This ensures that discoveries from previous runs are incorporated into
        # the gen model's knowledge before starting a new search. After training,
        # the discoveries are flagged as applied so they won't be re-trained on.
        if self._gen_mgr is not None:
            try:
                n_unapplied = self.db.count_unapplied()
                if n_unapplied > 0:
                    self._log(f"[TrainFirst] {n_unapplied} unapplied discoveries in DB — "
                              f"training gen model before search")
                    unapplied = self.db.get_unapplied_discoveries(limit=200)
                    solutions = []
                    applied_ids = []
                    for d in unapplied:
                        meta = d.get("metadata", {}) or {}
                        problem = meta.get("problem", "")
                        answer = meta.get("model_answer", "")
                        if problem and answer:
                            solutions.append({
                                "input": problem,
                                "output": answer,
                                "score": d.get("score", 0),
                            })
                        applied_ids.append(d["id"])
                    if solutions:
                        self._log(f"[TrainFirst] fine-tuning on {len(solutions)} solutions")
                        try:
                            stats = self._gen_mgr.fine_tune_on_solutions(solutions)
                            self._log(f"[TrainFirst] done: "
                                      f"{stats.get('n_examples', 0)} examples, "
                                      f"final_loss={stats.get('loss_history', [0])[-1]:.4f}")
                        except Exception as e:
                            self._log(f"[TrainFirst] fine-tune failed: {e}")
                    # Mark all unapplied discoveries as applied (even those
                    # without problem/answer text — they don't need training)
                    if applied_ids:
                        self.db.mark_applied(applied_ids)
                        self._log(f"[TrainFirst] marked {len(applied_ids)} discoveries as applied")
                else:
                    self._log("[TrainFirst] no unapplied discoveries — starting fresh search")
            except Exception as e:
                self._log(f"[TrainFirst] error checking unapplied: {e}")

        try:
            for gen in range(self.cfg.generations):
                self.generation = gen

                # Clear CUDA cache periodically to prevent fragmentation OOM
                if gen > 0 and gen % 5 == 0 and self.device.type == "cuda":
                    torch.cuda.empty_cache()

                # ── Novelty search: update phase (novelty ↔ quality pulsation) ──
                if self.cfg.enable_novelty and gen > 0:
                    self.novelty.update_phase(self.archive.best_score)
                    if gen == 1 or self.novelty.phase_counter == 1:
                        self._log(f"  [Novelty] {self.novelty.status()}")

                if self.cfg.time_budget_s > 0:
                    elapsed = time.time() - self.start_time
                    if elapsed > self.cfg.time_budget_s:
                        self._log(f"Time budget exceeded ({elapsed:.0f}s)")
                        break

                # ── Convergence detection: stop early if no improvement ──
                if self.cfg.convergence_patience > 0 and gen > 0:
                    if self.archive.best_score > self._prev_best_score:
                        self._convergence_counter = 0
                        self._prev_best_score = self.archive.best_score
                    else:
                        self._convergence_counter += 1
                        if self._convergence_counter >= self.cfg.convergence_patience:
                            self._log(
                                f"Converged: no improvement for "
                                f"{self.cfg.convergence_patience} generations "
                                f"(best={self.archive.best_score:.4f})")
                            # ── Dynamic domain spawning ──
                            # When a domain converges, spawn a refinement
                            # domain that narrows the search around the best
                            # config. This enables infinite-depth search.
                            if self.cfg.enable_refinement:
                                self._spawn_refinement()
                            break

                # ── PHASE 1: GENERATE (batched GPU forward) ──
                t0 = time.time()
                candidates = self.population.generate(n_per_gen=1)
                n_generated = len(candidates)
                t_gen = time.time() - t0

                # ── PHASE 1.5: DOMAIN-SPACE ELITE MUTANTS ──
                # Some domains (e.g. program synthesis) can mutate elite
                # configs directly in their own representation — a much
                # stronger operator than generator-weight noise when the
                # decode map is rugged. Opt-in via domain.propose_elite_mutants.
                if self.cfg.elite_mutants > 0 and gen > 0:
                    _mut_hook = getattr(self.domain, "propose_elite_mutants", None)
                    if _mut_hook is not None:
                        try:
                            for mc in _mut_hook(self.archive,
                                                self.cfg.elite_mutants, gen) or []:
                                candidates.append({"params": None,
                                                   "template_config": mc,
                                                   "generator_idx": -1})
                        except Exception as e:
                            self._log(f"  elite-mutant hook failed (gen {gen}): {e}")

                # ── PHASE 2: FILTER (surrogate on GPU + epsilon-greedy exploration) ──
                t0 = time.time()
                neural_cands = [c for c in candidates if c.get("params") is not None]
                template_cands = [c for c in candidates if c.get("template_config") is not None]

                n_eval = max(self.cfg.min_evaluate,
                             min(self.cfg.max_evaluate, n_generated // self.cfg.filter_ratio))

                n_template_eval = len(template_cands)
                n_neural_eval = max(0, n_eval - n_template_eval)

                if neural_cands and n_neural_eval > 0:
                    # Epsilon-greedy: fraction of eval budget goes to random candidates
                    # Novelty search: during novelty phase, increase random exploration
                    # to discover new behavioral regions
                    if self.cfg.enable_novelty and self.novelty.phase == "novelty":
                        epsilon = max(0.1, 0.5 - gen * 0.005)
                    else:
                        epsilon = max(0.1, 0.3 - gen * 0.005)
                    n_random = max(1, int(n_neural_eval * epsilon))
                    n_surrogate = n_neural_eval - n_random

                    # Surrogate-filtered selection
                    param_tensor = torch.stack([c["params"] for c in neural_cands]).to(self.device)
                    top_indices = self.surrogate.filter_top_k(param_tensor, n_surrogate)
                    selected_neural = [neural_cands[i] for i in top_indices.tolist()]

                    # Random selection (exploration)
                    remaining = [i for i in range(len(neural_cands)) if i not in top_indices.tolist()]
                    if remaining and n_random > 0:
                        random_indices = np.random.choice(remaining,
                                                          size=min(n_random, len(remaining)),
                                                          replace=False)
                        selected_neural += [neural_cands[i] for i in random_indices]
                else:
                    selected_neural = []

                selected = selected_neural + template_cands
                t_filter = time.time() - t0

                # ── PHASE 3: SCORE (domain evaluation, GPU+CPU parallel) ──
                t0 = time.time()
                scores = []
                behavioral_list = []
                metadata_list = []
                configs = []

                # ── Dedup filter: remove configs we've already evaluated ──
                # This happens BEFORE decoding/eval to save compute.
                unique_selected = []
                n_dupes = 0
                for cand in selected:
                    if "params" in cand and cand["params"] is not None:
                        # Dedup by full params tensor (rounded to 4 decimals)
                        p = cand["params"].detach().cpu().numpy()
                        key = ("p",) + tuple(round(float(x), 4) for x in p.flatten())
                    elif "template_config" in cand:
                        key = ("t", self._config_key(cand["template_config"]))
                    else:
                        key = ("id", id(cand))
                    if key in self._eval_cache:
                        n_dupes += 1
                        continue
                    self._eval_cache[key] = None  # placeholder until eval completes
                    unique_selected.append(cand)

                if n_dupes > 0 and self.cfg.verbose:
                    self._log(f"  Deduped {n_dupes}/{len(selected)} candidates")
                selected = unique_selected  # replace with deduped list

                # Decode all configs first — batch GPU→CPU transfer to avoid 50 sync points
                # Collect all param tensors, move to CPU in one call, then decode
                param_tensors = []
                param_indices = []
                for ci, cand in enumerate(selected):
                    if "params" in cand and cand["params"] is not None:
                        param_tensors.append(cand["params"])
                        param_indices.append(ci)

                if param_tensors:
                    # Single GPU→CPU transfer for all params at once
                    batched_params = torch.stack(param_tensors).detach().cpu()

                decoded = [None] * len(selected)
                for ci, cand in enumerate(selected):
                    if "params" in cand and cand["params"] is not None:
                        # Use pre-transferred CPU tensor
                        pi = param_indices.index(ci)
                        decoded[ci] = self.domain.decode(batched_params[pi])
                    else:
                        tc = cand["template_config"]
                        try:
                            config = self.domain.decode(self.domain.encode(tc))
                            config.update(tc)
                        except Exception:
                            # Encode failed (tc has missing keys for this domain).
                            # Decode a default config from mid-range params, then
                            # overlay the template's keys on top. This ensures all
                            # required keys are present.
                            try:
                                import torch as _t
                                default_params = _t.full(
                                    (self.domain.output_dim(),), 0.5,
                                    device=self.domain.device)
                                config = self.domain.decode(default_params)
                                config.update(tc)
                            except Exception:
                                config = tc
                        decoded[ci] = config

                if self.cfg.enable_compute_split and hasattr(self.domain, "set_gen_model"):
                    # ── Compute split path: foreground gen model + background checker ──
                    # Gen model inference runs on GPU (foreground).  Checker
                    # scoring (LLM-as-judge) is submitted to the background
                    # thread pool so it overlaps with the next batch.
                    results, configs = self._compute_split_eval(decoded)
                elif self.cpu_pool is not None and len(decoded) >= 8:
                    # Split: GPU gets 40%, CPU workers get 60% (CPU has more cores)
                    n_cpu = int(len(decoded) * 0.6)
                    n_gpu = len(decoded) - n_cpu
                    gpu_configs = decoded[:n_gpu]
                    cpu_configs = decoded[n_gpu:]

                    # Submit CPU batch asynchronously (non-blocking, chunked)
                    chunksize = max(1, len(cpu_configs) // (self.n_cpu_workers * 2))
                    cpu_result_async = self.cpu_pool.map_async(
                        _cpu_eval, cpu_configs, chunksize=chunksize,
                    )

                    # Run GPU batch with thread pool + CUDA streams for overlap
                    gpu_results = self._gpu_batch_eval(gpu_configs)

                    # Wait for CPU workers to finish
                    cpu_results = cpu_result_async.get()

                    # Re-merge in order
                    results = gpu_results + cpu_results
                    configs = gpu_configs + cpu_configs
                else:
                    # GPU-only: use threaded batch eval for kernel overlap
                    results = self._gpu_batch_eval(decoded)
                    configs = decoded

                # ── Collect results (no post-eval dedup needed — pre-filtered) ──
                for config, result in zip(configs, results):
                    scores.append(result["score"])
                    behavioral_list.append(result.get("behavioral", (0,)))
                    metadata_list.append(result.get("metadata", {}))

                    self.all_results.append({
                        "generation": gen,
                        "config": config,
                        "score": result["score"],
                        "behavioral": result.get("behavioral", (0,)),
                        "metadata": result.get("metadata", {}),
                    })

                # Cap results to prevent OOM on long runs
                if len(self.all_results) > self.max_results:
                    self.all_results = self.all_results[-self.max_results:]

                t_score = time.time() - t0

                # ── PHASE 4: TRAIN (GPU updates) ──
                t0 = time.time()

                # Update archive + save new discoveries to DB
                new_discoveries = 0
                new_discovery_list = []
                for config, score, behavioral, metadata in zip(configs, scores, behavioral_list, metadata_list):
                    # Novelty search: compute novelty + track behavioral
                    novelty_score = 0.0
                    if self.cfg.enable_novelty:
                        novelty_score = self.novelty.compute_novelty(behavioral)
                        self.novelty.add_behavioral(behavioral)

                    # Novelty-aware archive acceptance
                    if self.cfg.enable_novelty:
                        # During novelty phase, accept novel configs even if
                        # not the best in their cell (fills diverse cells)
                        if self.novelty.phase == "novelty" and novelty_score > self.novelty.cfg.min_novelty:
                            added = self.archive.add(config, score, behavioral, gen)
                        else:
                            added = self.archive.add(config, score, behavioral, gen)
                    else:
                        added = self.archive.add(config, score, behavioral, gen)
                    if added:
                        new_discoveries += 1
                        self.discoveries.append({
                            "generation": gen,
                            "config": config,
                            "score": score,
                            "behavioral": behavioral,
                        })
                        new_discovery_list.append({
                            "generation": gen,
                            "config": config,
                            "score": score,
                            "behavioral": behavioral,
                            "metadata": metadata,
                        })

                # Save new discoveries to database (batched — flush every 10 gens)
                if new_discovery_list:
                    self._pending_discoveries.extend(new_discovery_list)
                    if (gen + 1) % 10 == 0 or gen == self.cfg.generations - 1:
                        self.db.save_discoveries(self.run_id, self.domain.name(),
                                                 self._pending_discoveries)
                        self._pending_discoveries = []

                # Update surrogate
                if neural_cands and selected_neural:
                    eval_params = torch.stack([c["params"] for c in selected_neural]).detach()
                    eval_scores = torch.tensor(
                        scores[:len(selected_neural)], dtype=torch.float32
                    )
                    try:
                        self.surrogate.train(eval_params, eval_scores)
                    except Exception as e:
                        self._log(f"Surrogate training skipped (gen {gen}): {e}")
                    self.surrogate.generation = gen

                # Update generators (REINFORCE) — pass context
                self.trainer.update(selected, scores, self.population.context)

                # Evolve population
                gen_indices = [c.get("generator_idx", -1) for c in selected]
                mutated = self.population.evolve(scores, gen_indices)

                # Notify trainer of mutations so it can reset optimizer state
                for idx in mutated:
                    self.trainer.notify_mutation(idx)

                # Update context: UCB-selected parent (not just best)
                # 70% UCB selection, 30% best (exploit known good)
                if np.random.random() < 0.7:
                    parent = self.archive.sample_elite_ucb(current_gen=gen, c=1.0)
                else:
                    parent = self.archive.best_entry
                if parent is None:
                    parent = self.archive.best_entry
                if parent:
                    parent_params = self.domain.encode(parent.config)
                    if parent_params is not None:
                        self.population.update_context(parent_params)

                t_train = time.time() - t0

                # Incremental generator/surrogate checkpoint every 10 gens
                if (gen + 1) % 10 == 0:
                    self.db.save_generators(self.run_id, self.domain.name(),
                                            self.population.batched_gen)
                    self.db.save_surrogate(self.run_id, self.domain.name(), self.surrogate)

                # ── LOG ──
                # Always log gen 0, final gen, and every log_every gens
                is_last_gen = (gen == self.cfg.generations - 1)
                if gen == 0 or gen % self.cfg.log_every == 0 or is_last_gen:
                    # Track phase timings for GPU utilization analysis
                    phase_info = (f"gen={t_gen:.3f}s filter={t_filter:.3f}s "
                                  f"score={t_score:.2f}s train={t_train:.3f}s")
                    self._log(
                        f"{phase_info} | "
                        f"evaluated={len(selected)}/{n_generated} | "
                        f"new={new_discoveries} | "
                        f"{self.archive.summary()}"
                    )
                    # Detect if score phase dominates (good = GPU busy)
                    # vs gen/filter phase dominates (bad = GPU idle on Python)
                    total_t = t_gen + t_filter + t_score + t_train
                    if total_t > 0 and t_score / total_t < 0.3 and gen > 2:
                        # Score phase < 30% of total = GPU underutilized.
                        # Increase eval batch to keep GPU busy longer.
                        old_max = self.cfg.max_evaluate
                        self.cfg.max_evaluate = min(old_max * 2, 200)
                        if self.cfg.max_evaluate != old_max:
                            self._log(f"  [GPUKeepBusy] max_eval {old_max}→{self.cfg.max_evaluate} "
                                     f"(score phase only {t_score/total_t*100:.0f}% of time)")

                # ── ADAPTIVE POPULATION SIZING ──
                # Grow population when plateauing (more explorers needed),
                # shrink when steadily improving (efficiency gain).
                if self.cfg.adaptive_population and gen > 0:
                    if self.archive.best_score > self._prev_best_score + 1e-6:
                        self._pop_stable_count += 1
                        if self._pop_stable_count >= 3:
                            # Steady improvement → shrink (save compute)
                            new_n = max(self.cfg.pop_min,
                                        int(self.cfg.n_generators * (1 - self.cfg.pop_shrink_rate)))
                            if new_n != self.cfg.n_generators:
                                self.cfg.n_generators = new_n
                                self._log(f"  [AdaptivePop] shrinking to {new_n} (steady improvement)")
                            self._pop_stable_count = 0
                    else:
                        self._pop_plateau_count += 1
                        if self._pop_plateau_count >= 2:
                            # Plateau → grow (need more exploration)
                            new_n = min(self.cfg.pop_max,
                                        int(self.cfg.n_generators * (1 + self.cfg.pop_growth_rate)))
                            if new_n != self.cfg.n_generators:
                                self.cfg.n_generators = new_n
                                self._log(f"  [AdaptivePop] growing to {new_n} (plateau detected)")
                            self._pop_plateau_count = 0

                # ── COMPUTE SPLIT: gen model lifecycle (grow/shrink/fine-tune) ──
                if self.cfg.enable_compute_split and self._gen_mgr is not None:
                    # Record this round's performance
                    best_score = self.archive.best_score
                    gm = getattr(self._gen_mgr, "_model", None)
                    gm_size = None
                    if gm is not None:
                        if hasattr(gm, "get_size_config"):
                            gm_size = gm.get_size_config()
                        elif hasattr(gm, "param_count"):
                            gm_size = {"params": gm.param_count()}
                    self._gen_mgr.record_round(self.domain.name(), best_score, gm_size)

                    # Grow if plateauing
                    if self._gen_mgr.should_grow():
                        self._log(f"  [GenModel] growing (plateau {self._gen_mgr._plateau_count} rounds)")
                        self._gen_mgr.grow()
                        if gm is not None and hasattr(self.domain, "set_gen_model"):
                            self.domain.set_gen_model(self._gen_mgr._model)
                    elif self._gen_mgr.should_shrink():
                        self._log("  [GenModel] shrinking (overperforming)")
                        self._gen_mgr.shrink()
                        if hasattr(self.domain, "set_gen_model"):
                            self.domain.set_gen_model(self._gen_mgr._model)

                    # Fine-tune on successful solutions every 5 rounds
                    if (gen + 1) % 5 == 0 and self._gen_mgr is not None:
                        solutions = []
                        for d in self.discoveries[-20:]:
                            meta = d.get("metadata", {})
                            problem = meta.get("problem", "")
                            answer = meta.get("model_answer", "")
                            if problem and answer:
                                solutions.append({"prompt": problem, "output": answer})
                        if solutions and self._gen_mgr is not None:
                            self._log(f"  [GenModel] fine-tuning on {len(solutions)} solutions")
                            try:
                                self._gen_mgr.fine_tune_on_solutions(solutions)
                            except Exception as e:
                                self._log(f"  [GenModel] fine-tune failed: {e}")

            # Final summary
            elapsed = time.time() - self.start_time
            self._log(f"Done: {self.generation+1} generations, "
                      f"{len(self.all_results)} evaluations, "
                      f"{len(self.discoveries)} discoveries, "
                      f"{elapsed:.1f}s total")

            # Save final state to database
            # Flush any remaining pending discoveries
            if self._pending_discoveries:
                n_saved, n_deduped = self.db.save_discoveries(
                    self.run_id, self.domain.name(),
                    self._pending_discoveries)
                if n_deduped > 0:
                    self._log(f"DB dedup: saved {n_saved}, skipped {n_deduped} duplicates")
                self._pending_discoveries = []

            # Build final results dict (includes refinement domain info)
            results = {
                "generations": self.generation + 1,
                "total_evaluations": len(self.all_results),
                "discoveries": len(self.discoveries),
                "best_score": self.archive.best_score,
                "best_config": self.archive.best_entry.config if self.archive.best_entry else None,
                "device": str(self.device),
                "pending_refinements": [d.name() for d in self.pending_refinements],
                "spawned_domains": self.domain_factory.get_spawned_names(),
            }
            self.db.save_run(self.run_id, self.domain.name(),
                             {"n_generators": self.cfg.n_generators,
                              "filter_ratio": self.cfg.filter_ratio,
                              "generations": self.cfg.generations},
                             results, self.start_time)
            self.db.save_generators(self.run_id, self.domain.name(),
                                    self.population.batched_gen)
            self.db.save_surrogate(self.run_id, self.domain.name(), self.surrogate)

            # ── Canonical save: update permanent knowledge ONLY if this run
            # produced new discoveries. If the run found nothing new (all
            # configs were dupes or archive didn't improve), skip the save
            # to avoid overwriting canonical weights with identical ones.
            # Skip canonical save for Generic domains (heuristic evaluator,
            # not real performance data — would pollute cross-domain transfer)
            is_generic = "Generic_" in self.domain.name()
            best = self.archive.best_score
            n_new = len(self.discoveries)
            canonical_best = self.db.get_canonical_best_score(self.domain.name())
            if best > float('-inf') and n_new > 0 and not is_generic:
                # Only save if we actually found new discoveries AND beat
                # the canonical best (the DB method checks the score)
                updated_gen = self.db.save_canonical_generators(
                    self.domain.name(), self.population.batched_gen,
                    best, self.run_id)
                updated_surr = self.db.save_canonical_surrogate(
                    self.domain.name(), self.surrogate, best, self.run_id)
                if updated_gen or updated_surr:
                    self._log(f"Updated canonical knowledge for '{self.domain.name()}' "
                              f"(best={best:.4f}) — future runs will start from here")
            elif best > float('-inf') and n_new == 0:
                canon_str = f"{canonical_best:.4f}" if canonical_best is not None else "none"
                self._log(f"No new discoveries for '{self.domain.name()}' — "
                          f"canonical unchanged (best={best:.4f}, "
                          f"canonical={canon_str})")

            self._log(f"Saved to database: {self.cfg.db_path} (run_id={self.run_id})")

            results = {
                "generations": self.generation + 1,
                "total_evaluations": len(self.all_results),
                "discoveries": len(self.discoveries),
                "archive_coverage": self.archive.coverage(),
                "best_score": self.archive.best_score,
                "best_config": self.archive.best_entry.config if self.archive.best_entry else None,
                "pareto_front": [
                    {"config": e.config, "score": e.score, "behavioral": e.behavioral,
                     "generation": e.generation, "metadata": e.metadata}
                    for e in self.archive.get_pareto_front()
                ],
                "time_s": elapsed,
                "all_results": self.all_results,
                "discoveries_list": self.discoveries,
                "archive": self.archive,
                "device": str(self.device),
                "run_id": self.run_id,
                "warm_started": self._warm_started,
                "pending_refinements": [d.name() for d in self.pending_refinements],
                "spawned_domains": self.domain_factory.get_spawned_names(),
            }
        finally:
            # Clean up CPU worker pool
            if self.cpu_pool is not None:
                self.cpu_pool.close()
                self.cpu_pool.join()
                self.cpu_pool = None
            # Clean up checker thread pool
            if self._checker_pool is not None:
                self._checker_pool.shutdown(wait=False, cancel_futures=True)
                self._checker_pool = None

        return results

    def guide(self, direction: str = "explore",
              seed_configs: list[dict] | None = None,
              constraints: dict | None = None,
              time_budget: str | float | None = None):
        """Steer the search mid-run (applies on next generation)."""
        if direction == "explore":
            self.cfg.exploration = 0.8
            self.cfg.mutation_rate = 0.3
        elif direction == "exploit":
            self.cfg.exploration = 0.1
            self.cfg.mutation_rate = 0.02

        if seed_configs:
            for config in seed_configs:
                result = self.domain.evaluate(config)
                self.archive.add(config, result["score"],
                                 result.get("behavioral", (0,)), self.generation)

        if time_budget:
            if isinstance(time_budget, str):
                import re
                m = re.match(r"(\d+(?:\.\d+)?)\s*(s|m|h)", time_budget)
                if m:
                    val, unit = float(m.group(1)), m.group(2)
                    self.cfg.time_budget_s = val * {"s": 1, "m": 60, "h": 3600}[unit]
            else:
                self.cfg.time_budget_s = time_budget
