"""Checkpoint merge and evolutionary-merge mixin for ForgeEngine."""
from pathlib import Path

from .engine_common import *  # noqa: F403
from .errors import ConfigurationError  # noqa: F401
from .engine_common import (  # noqa: F401
    _CKPT_CACHE_MAX,
    _DEFAULT_CPU_MEMORY_BYTES,
    _DEFAULT_EOS_TOKEN_IDS,
    _QWEN_TOKENIZER_PATH,
    _QWEN_VOCAB,
    _checkpoint_metadata_cache,
    _checkpoint_size_cache,
    _ckpt_cache_lock,
    _map_gguf_to_forge,
    _min_k_filter,
    _ScalingModelAdapter,
    _tokenizer_for_vocab,
    logger,
)


class _MergingMixin:
    @staticmethod
    def _load_state_dict_cpu(path: str) -> dict:
        """Load a checkpoint's tensors to CPU (for merge/evolve operations)."""
        from forge.checkpoint_io import load_checkpoint
        state = load_checkpoint(path, map_location="cpu")
        return {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}

    def _swap_weights(self, state: dict[str, torch.Tensor]) -> None:
        """Hot-swap model weights in-place from a CPU state dict.

        Loads the given state dict into the resident model using
        ``load_state_dict(assign=True)`` — no model reconstruction, just
        pointer swaps. Used as the fitness-evaluation primitive for
        evolutionary merging: swap in a candidate, benchmark, swap back.

        The model must already be built with the same architecture (same
        config_name as the parents). All parents in an evolutionary run
        MUST share the same architecture for this to work.
        """
        # Serialise against in-flight generation: load_state_dict(assign=True)
        # rebinding mid-forward corrupts the active decode.
        with self._gen_lock:
            dev_state = {k: (v.to(self.device) if not v.is_meta else v)
                         for k, v in state.items()}
            missing, unexpected = self.model.load_state_dict(
                dev_state, strict=False, assign=True)
            if dev_state and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

    def merge_checkpoints(
        self,
        parents: list[str],
        method: str = "blockwise_crossover",
        out_path: str | None = None,
        config_name: str | None = None,
        load_result: bool = False,
        **kwargs,
    ) -> str:
        """One-shot merge of 2+ checkpoint paths → saved merged checkpoint.

        Wraps the merge functions from ``research.merge_models.py``. The
        merged checkpoint is saved to ``out_path`` (or a temp file) and the
        path is returned. If ``load_result=True``, the merged weights are
        also hot-swapped into this engine.

        Args:
            parents: list of checkpoint paths (>=2 for crossover, >=1 for
                mutation).
            method: one of:
                - "blockwise_crossover" (2 parents, block-level splice)
                - "block_random_crossover" (2 parents, per-block random)
                - "uniform_crossover" (2 parents, per-tensor random)
                - "gaussian_mutation" (1 parent, Gaussian noise)
                - "quant_perturb" (1 parent, quantization-scale mutation)
                - "block_swap" (2 parents: recipient + donor)
                - "slerp", "linear", "ties", "dare", "svd", "task_arith"
                  (legacy merge methods from merge_models.py)
            out_path: where to save the merged checkpoint. If None, uses
                ``data/merged/<method>_<timestamp>.safetensors``.
            config_name: config name for the parents (used only if
                load_result=True). Defaults to the engine's current config.
            load_result: if True, hot-swap the merged weights into this
                engine after saving.
            **kwargs: method-specific params (e.g. split_block, p, sigma,
                rate, t, density, drop_rate, rank_ratio, seed).

        Returns:
            Path to the saved merged checkpoint.
        """
        from forge.checkpoint_io import save_checkpoint
        from research.merge_models import (
            _task_vectors,
            crossover_block_random,
            crossover_blockwise,
            crossover_uniform,
            merge_dare,
            merge_linear,
            merge_slerp,
            merge_svd,
            merge_task_arith,
            merge_ties,
            mutate_block_swap,
            mutate_gaussian,
            mutate_quant_perturb,
        )
        from research.paths import DATA_DIR

        if len(parents) < 1:
            raise ConfigurationError("merge_checkpoints needs >=1 parent")

        states = [self._load_state_dict_cpu(p) for p in parents]
        seed = kwargs.pop("seed", 0)

        method_map = {
            "blockwise_crossover": lambda: crossover_blockwise(
                states[0], states[1], seed=seed, **kwargs),
            "block_random_crossover": lambda: crossover_block_random(
                states[0], states[1], seed=seed, **kwargs),
            "uniform_crossover": lambda: crossover_uniform(
                states[0], states[1], seed=seed, **kwargs),
            "gaussian_mutation": lambda: mutate_gaussian(
                states[0], seed=seed, **kwargs),
            "quant_perturb": lambda: mutate_quant_perturb(
                states[0], seed=seed, **kwargs),
            "block_swap": lambda: mutate_block_swap(
                states[0], states[1], seed=seed, **kwargs),
            "slerp": lambda: merge_slerp(
                states[0], states[1], t=kwargs.pop("t", 0.5)),
            "linear": lambda: merge_linear(states, kwargs.pop("weights", None)),
            "ties": lambda: merge_ties(
                states[0], [_task_vectors(s, states[0]) for s in states[1:]],
                density=kwargs.pop("density", 0.5)),
            "dare": lambda: merge_dare(
                states[0], [_task_vectors(s, states[0]) for s in states[1:]],
                drop_rate=kwargs.pop("drop_rate", 0.1), seed=seed),
            "svd": lambda: merge_svd(
                states[0], [_task_vectors(s, states[0]) for s in states[1:]],
                rank_ratio=kwargs.pop("rank_ratio", 0.5),
                scales=kwargs.pop("weights", None)),
            "task_arith": lambda: merge_task_arith(
                states[0], [_task_vectors(s, states[0]) for s in states[1:]],
                scales=kwargs.pop("weights", None)),
        }
        if method not in method_map:
            raise ConfigurationError(
                f"Unknown merge method: {method}. Valid: {list(method_map)}")

        merged = method_map[method]()

        if out_path is None:
            import time as _t
            out_path = str(Path(DATA_DIR) / "merged" /
                           f"{method}_{int(_t.time())}.safetensors")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(merged, out_path)
        self._log(f"Merge ({method}): {len(parents)} parents → {out_path} "
                  f"({len(merged)} tensors)")

        if load_result:
            self._swap_weights(merged)
            self._log("Merge result hot-swapped into engine")

        return out_path

    def evolve_merge(
        self,
        parents: list[str],
        fitness_fn=None,
        n_generations: int = 5,
        population_size: int | None = None,
        crossover: str = "blockwise",
        crossover_kwargs: dict | None = None,
        mutation: str = "gaussian",
        mutation_kwargs: dict | None = None,
        mutation_rate: float = 0.5,
        elitism: int = 1,
        out_dir: str | None = None,
        seed: int = 0,
        benchmark_prompt: str = "The quick brown fox jumps over the lazy dog.",
        benchmark_tokens: int = 32,
        restore_original: bool = True,
        verbose: bool = True,
        selection: str = "tournament",
        selection_kwargs: dict | None = None,
        fitness_scaling: str = "none",
        progress_bonus: float = 0.0,
        diversity_bonus: float = 0.0,
        adaptive_mutation: bool = False,
        hall_of_fame_size: int = 0,
        convergence_patience: int = 0,
    ) -> dict:
        """Evolutionary model merging — "sexually reproduce" LLM checkpoints.

        Runs the GENOME-style evolutionary loop (crossover → mutation →
        selection → succession) over a population of checkpoint paths,
        using this engine as the fitness evaluator. The best offspring is
        saved to disk and its path returned.

        **Sophisticated candidate selection** (``selection`` parameter):
          - "tournament": k-way tournament (classic, robust, default)
          - "rank": linear ranking with selection pressure (outlier-resistant)
          - "roulette": fitness-proportionate (fast, needs non-negative fitness)
          - "diversity": novelty-quality hybrid (prevents premature convergence)

        **Score rewarding**:
          - ``fitness_scaling``: "none", "sigma" (linear sigma scaling),
            "rank" (rank-based). Prevents single-individual dominance.
          - ``progress_bonus``: rewards offspring that beat parent fitness.
          - ``diversity_bonus``: rewards individuals far from population centroid.
          - ``adaptive_mutation``: dynamically adjusts mutation_rate based on
            population diversity (increases on stagnation, decreases on convergence).
          - ``hall_of_fame_size``: maintains best-ever individuals as elite donors.
          - ``convergence_patience``: early stop if no improvement for N gens.

        **VRAM budget**: All weight recombination happens in CPU RAM. The
        GPU is only used to evaluate one candidate at a time via in-place
        weight hot-swap (``_swap_weights``). For a 1.2B model, each
        candidate eval is ~2-3s; population_size=8 × n_generations=5 =
        ~40 evals ≈ 2 minutes. No additional VRAM beyond the resident model.

        **Fitness function**: If ``fitness_fn`` is None, uses
        ``engine.benchmark(benchmark_prompt, max_new_tokens=benchmark_tokens)``
        and returns ``tokens_per_sec`` (higher = faster = better). Pass a
        custom ``fitness_fn(state_dict) -> float`` for quality-based
        fitness (e.g. perplexity, benchmark accuracy — negate for
        minimization). The custom fn receives a CPU state dict and is
        responsible for any GPU loading (or call ``engine._swap_weights``).

        Args:
            parents: list of checkpoint paths (>=2). These form the initial
                population. All MUST share the same architecture/config.
            fitness_fn: callable(cpu_state_dict) -> float. If None, uses
                benchmark speed (tok/s). Higher = better.
            n_generations: number of evolutionary generations.
            population_size: target pop size per gen. None = len(parents).
            crossover: "blockwise", "block_random", or "uniform".
            crossover_kwargs: extra kwargs for the crossover function.
            mutation: "gaussian", "quant_perturb", or "block_swap".
            mutation_kwargs: extra kwargs for the mutation function.
            mutation_rate: probability each offspring is mutated.
            elitism: number of top individuals carried forward unchanged.
            out_dir: directory for saving per-generation best + final best.
                If None, uses ``data/evolved/<timestamp>/``.
            seed: base RNG seed.
            benchmark_prompt: prompt for the default benchmark fitness.
            benchmark_tokens: max_new_tokens for the default benchmark fitness.
            restore_original: if True, restore the engine's original weights
                after evolution (the engine is used as an evaluator, not
                mutated in place). Set False to keep the best weights loaded.
            verbose: print per-generation progress.

        Returns:
            dict from ``research.merge_models.evolve()``:
            - "best": best state dict (CPU)
            - "best_fitness": float
            - "best_path": path to saved best checkpoint (if out_dir set)
            - "history": list of per-gen fitness stats
            - "final_population": list of state dicts
        """
        from forge.checkpoint_io import save_checkpoint
        from research.merge_models import evolve
        from research.paths import DATA_DIR

        if len(parents) < 2:
            raise ConfigurationError(
                "evolve_merge needs >=2 parent checkpoints to breed")

        self._require_awake()
        self._log(f"Evolutionary merge: {len(parents)} parents, "
                  f"{n_generations} generations, crossover={crossover}, "
                  f"mutation={mutation}")

        # Save original weights so we can restore the engine after evolution
        original_state = None
        if restore_original:
            original_state = {
                k: v.detach().clone().to("cpu")
                for k, v in self.model.state_dict().items()
            }

        # Build the fitness function
        if fitness_fn is None:
            def fitness_fn(state: dict) -> float:
                """Default fitness: generation speed (tok/s). Higher = better."""
                try:
                    self._swap_weights(state)
                    result = self.benchmark(
                        benchmark_prompt, max_new_tokens=benchmark_tokens,
                        n_runs=1)
                    return result["tokens_per_sec"]
                except Exception as e:
                    self._log(f"Fitness eval failed: {e}", level="warn")
                    return 0.0

        # Load initial population to CPU
        population = [self._load_state_dict_cpu(p) for p in parents]

        # Output directory
        if out_dir is None:
            import time as _t
            out_dir = str(Path(DATA_DIR) / "evolved" / f"run_{int(_t.time())}")

        result = evolve(
            population,
            fitness_fn,
            n_generations=n_generations,
            population_size=population_size,
            crossover=crossover,
            crossover_kwargs=crossover_kwargs,
            mutation=mutation,
            mutation_kwargs=mutation_kwargs,
            mutation_rate=mutation_rate,
            elitism=elitism,
            seed=seed,
            save_fn=save_checkpoint,
            out_dir=out_dir,
            verbose=verbose,
            selection=selection,
            selection_kwargs=selection_kwargs,
            fitness_scaling=fitness_scaling,
            progress_bonus=progress_bonus,
            diversity_bonus=diversity_bonus,
            adaptive_mutation=adaptive_mutation,
            hall_of_fame_size=hall_of_fame_size,
            convergence_patience=convergence_patience,
        )

        self._log(f"Evolution complete: best_fitness={result['best_fitness']:.4f}, "
                  f"best_path={result['best_path']}")

        # Restore original weights or load the best
        if restore_original and original_state is not None:
            self._swap_weights(original_state)
            self._log("Restored original engine weights after evolution")
        elif result["best"] is not None:
            self._swap_weights(result["best"])
            self._log("Loaded best evolved weights into engine")

        return result

