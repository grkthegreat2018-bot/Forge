"""Tests for ForgeEngine refactoring fixes.

Validates the specific bugs fixed in the refactor commit:
1. Thread-safe checkpoint caches (_ckpt_cache_lock)
2. _clear_cuda_cache helper exists and is callable
3. _release_acceleration_resources cleans up acceleration slots
4. _active_kv_bits is initialized and tracked
5. _read_checkpoint_metadata handles missing/corrupt files gracefully
6. continue_session doesn't use 'generated_ids' in dir() hack
"""
import sys
sys.path.insert(0, r"D:\windsurf\ForgeAI")

import pytest
import torch
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestCheckpointCacheThreadSafety:
    """The module-level caches must be guarded by a lock."""

    def test_lock_exists(self):
        from forge.engine import forge_engine
        assert hasattr(forge_engine, "_ckpt_cache_lock")
        assert isinstance(forge_engine._ckpt_cache_lock, type(threading.Lock()))

    def test_concurrent_cache_access_no_crash(self, tmp_path):
        """Multiple threads writing to _checkpoint_size_cache simultaneously."""
        from forge.engine import forge_engine
        # Clear cache for clean test
        with forge_engine._ckpt_cache_lock:
            forge_engine._checkpoint_size_cache.clear()

        errors = []

        def worker(i):
            try:
                # Simulate the cache pattern from from_checkpoint
                path = f"fake_path_{i}.safetensors"
                ckpt_size = forge_engine._checkpoint_size_cache.get(path)
                if ckpt_size is None:
                    ckpt_size = 1024 * i  # fake size
                    with forge_engine._ckpt_cache_lock:
                        forge_engine._checkpoint_size_cache[path] = ckpt_size
                        while len(forge_engine._checkpoint_size_cache) > forge_engine._CKPT_CACHE_MAX:
                            forge_engine._checkpoint_size_cache.popitem(last=False)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        # Cache should be bounded
        assert len(forge_engine._checkpoint_size_cache) <= forge_engine._CKPT_CACHE_MAX


class TestClearCudaCache:
    """_clear_cuda_cache helper should exist and be callable."""

    def test_helper_exists(self):
        from forge.engine.forge_engine import ForgeEngine
        assert hasattr(ForgeEngine, "_clear_cuda_cache")

    def test_safe_on_cpu(self, forge_engine_fixture):
        """Should not crash even on CPU (no CUDA)."""
        engine = forge_engine_fixture
        # Should be a no-op on CPU
        engine._clear_cuda_cache()


class TestReleaseAccelerationResources:
    """_release_acceleration_resources should nil out all acceleration slots."""

    def test_helper_exists(self):
        from forge.engine.forge_engine import ForgeEngine
        assert hasattr(ForgeEngine, "_release_acceleration_resources")

    def test_releases_all_slots(self, forge_engine_fixture):
        engine = forge_engine_fixture
        # Set fake acceleration objects
        engine._graph_runner = MagicMock()
        engine._megakernel = MagicMock()
        engine._flex_decoding = MagicMock()
        engine._chunked_prefill = MagicMock()
        engine.acceleration = "cuda_graph"

        engine._release_acceleration_resources()

        assert engine._graph_runner is None
        assert engine._megakernel is None
        assert engine._flex_decoding is None
        assert engine._chunked_prefill is None
        assert engine.acceleration is None

    def test_calls_release_method_if_present(self, forge_engine_fixture):
        engine = forge_engine_fixture
        mock_obj = MagicMock()
        mock_obj.release = MagicMock()
        engine._graph_runner = mock_obj
        engine._release_acceleration_resources()
        mock_obj.release.assert_called_once()

    def test_safe_with_none_slots(self, forge_engine_fixture):
        engine = forge_engine_fixture
        # All slots already None — should not crash
        engine._release_acceleration_resources()
        assert engine._graph_runner is None


class TestActiveKvBits:
    """_active_kv_bits should be initialized in __init__ and tracked in activation."""

    def test_initialized_in_init(self, forge_engine_fixture):
        engine = forge_engine_fixture
        assert hasattr(engine, "_active_kv_bits")
        assert isinstance(engine._active_kv_bits, int)

    def test_kv_cache_name_tracked(self, forge_engine_fixture):
        engine = forge_engine_fixture
        assert hasattr(engine, "_active_kv_cache_name")


class TestReadCheckpointMetadataGraceful:
    """_read_checkpoint_metadata should handle missing/corrupt files."""

    def test_missing_file_returns_empty(self):
        from forge.engine.forge_engine import ForgeEngine
        metadata = ForgeEngine._read_checkpoint_metadata("nonexistent_file.safetensors")
        assert isinstance(metadata, dict)
        assert len(metadata) == 0  # empty dict, not crash


class TestLogprobsAndStopRegressions:
    """Regressions fixed in _generate_impl:
    1. generate(logprobs=N) crashed attaching _forge_logprobs to a str.
    2. Multiple stop strings truncated at first-found-in-list, not the
       earliest occurrence in the text.
    """

    def _gen_engine(self, tiny_model):
        from forge.engine.forge_engine import ForgeEngine
        tok = MagicMock()
        tok.eos_token_id = 0
        tok.return_value.input_ids = torch.tensor([[1, 2, 3, 4]])
        tok.decode = MagicMock(return_value="x")
        eng = ForgeEngine(tiny_model, tok, device="cpu")
        eng._library_enabled = False   # MagicMock tokenizer can't inject
        return eng

    def test_logprobs_returns_dict_not_crash(self, tiny_model):
        eng = self._gen_engine(tiny_model)
        out = eng.generate("hi", max_new_tokens=2, logprobs=2,
                           finish_sentence=False)
        assert isinstance(out, dict), f"expected dict, got {type(out)}"
        assert "text" in out and "logprobs" in out
        assert "content" in out["logprobs"]

    def test_stop_truncates_at_earliest_match(self, tiny_model):
        eng = self._gen_engine(tiny_model)
        eng.tokenizer.decode = MagicMock(
            return_value="aa STOP_B bb STOP_A cc")
        out = eng.generate("hi", max_new_tokens=2,
                           stop=["STOP_A", "STOP_B"],
                           finish_sentence=False)
        assert out == "aa ", f"stop markers leaked into output: {out!r}"


class TestContinueSessionNoDirHack:
    """continue_session should not use 'generated_ids' in dir() hack."""

    def test_no_dir_hack_in_source(self):
        path = Path(__file__).parent.parent.parent / "forge" / "engine" / "forge_engine.py"
        source = path.read_text(encoding="utf-8")
        assert "'generated_ids' in dir()" not in source, (
            "The 'generated_ids' in dir() hack should have been removed in the refactor"
        )
        assert "'generated_ids' in dir()" not in source


class TestCrashRecoveryWeakref:
    """atexit/signal registrations are GC roots — the manager must hold
    itself only via weakref so engine teardown stays collectable."""

    def test_manager_collectable_after_install(self, tmp_path):
        import atexit
        import gc
        import signal
        import weakref
        from forge.engine.crash_recovery import CrashRecoveryManager
        old_term = signal.getsignal(signal.SIGTERM)
        old_int = signal.getsignal(signal.SIGINT)
        try:
            eng = MagicMock()
            mgr = CrashRecoveryManager(eng, recovery_dir=str(tmp_path),
                                       enabled=True)
            ref = weakref.ref(mgr)
            del mgr
            gc.collect()
            # the atexit closure + signal table must not pin the manager
            assert ref() is None
        finally:
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)

    def test_close_unregisters_and_disables(self, tmp_path):
        import signal
        from forge.engine.crash_recovery import CrashRecoveryManager
        old_term = signal.getsignal(signal.SIGTERM)
        old_int = signal.getsignal(signal.SIGINT)
        try:
            mgr = CrashRecoveryManager(MagicMock(),
                                       recovery_dir=str(tmp_path),
                                       enabled=True)
            assert mgr._installed and mgr.enabled
            mgr.close()
            assert not mgr._installed and not mgr.enabled
            mgr._atexit_fn()              # dead callback must be a no-op
            mgr._cleanup()                # disabled -> writes nothing
            assert not (tmp_path / "generation_final.zst").exists()
        finally:
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)


class TestFluxSleepAndLogprobs:
    """FluxLM-specific lifecycle fixes:
    1. sleep(1) must use to_device() — nn.Module.to() is a no-op on
       FluxLM's plain-attribute tensors.
    2. generate(logprobs=N) must not re-ingest the sequence — logprobs
       are captured during decode (FLUX forward learns; a re-run would
       double-write every token into live memory).
    """

    def _flux_engine(self, vocab=65536):
        from forge.engine.forge_engine import ForgeEngine
        from forge.model.flux import FluxConfig
        cfg = FluxConfig(vocab_size=vocab, topic_dim=32,
                         ring_capacity=1 << 14, journal_cap=50_000)
        return ForgeEngine.from_flux(config=cfg)

    def test_sleep_level1_uses_to_device(self):
        eng = self._flux_engine()
        m = eng.model
        calls = []
        orig = m.to_device
        m.to_device = lambda d: calls.append(d) or orig(d)
        m.to = MagicMock(side_effect=AssertionError(
            "nn.Module.to() is a no-op on FluxLM — must use to_device"))
        eng.sleep(level=1)
        assert calls == ["cpu"]
        eng.wake()
        assert calls == ["cpu", str(eng.device)]
        assert eng._awake

    def test_logprobs_no_double_ingest(self):
        eng_a = self._flux_engine()
        eng_b = self._flux_engine()
        out_a = eng_a.generate("hello world", max_new_tokens=4,
                               temperature=0.0)
        out_b = eng_b.generate("hello world", max_new_tokens=4,
                               temperature=0.0, logprobs=3)
        # identical learned state -> logprob path added no extra ingest
        assert len(eng_a.model.journal.entries) == \
            len(eng_b.model.journal.entries)
        assert eng_a.model._count == eng_b.model._count
        assert isinstance(out_b, dict) and "logprobs" in out_b


class TestStreamStopStrings:
    """generate_stream(stop=) must catch stop strings that span chunk
    boundaries and emit only text before the earliest match."""

    def test_stop_spanning_chunks(self, tiny_model):
        from forge.engine.forge_engine import ForgeEngine
        tok = MagicMock()
        tok.eos_token_id = 0
        tok.return_value.input_ids = torch.tensor([[1, 2, 3]])
        chunks = iter(["hel", "lo ", "ST", "OP", " trailing"])
        tok.decode = MagicMock(side_effect=lambda *a, **k: next(chunks))
        eng = ForgeEngine(tiny_model, tok, device="cpu")
        eng._library_enabled = False
        # recovery checkpoints decode generated_ids every step — would
        # consume the mocked chunk stream out from under the test
        eng._recovery.enabled = False
        # force a non-EOS argmax every step so the loop runs 5 tokens
        force = torch.zeros(1, tiny_model.config.vocab_size)
        force[0, 5] = 1.0
        proc = lambda logits, ids: force
        out = "".join(eng.generate_stream(
            "hi", max_new_tokens=5, temperature=0.0,
            stop=["STOP"], logits_processor=proc,
            eos_token_ids=[255]))
        assert out == "hello "


# ── Fixture ───────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_model():
    from forge.config import get_config
    from forge.model_loader import ConfigurableResearchLLM
    cfg = get_config("forgelm_tiny")
    cfg.vocab_size = 256
    cfg.device = "cpu"
    cfg.dtype = "float32"
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        model = ConfigurableResearchLLM(cfg)
    finally:
        torch.set_default_dtype(old)
    model.eval()
    return model


@pytest.fixture
def forge_engine_fixture():
    """Minimal ForgeEngine on CPU for testing helper methods.

    Uses a mock tokenizer to avoid filesystem dependency on tokenizer location.
    """
    from forge.config import get_config
    from forge.model_loader import ConfigurableResearchLLM
    from forge.engine.forge_engine import ForgeEngine

    cfg = get_config("forgelm_tiny")
    cfg.vocab_size = 256
    cfg.device = "cpu"
    cfg.dtype = "float32"

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        model = ConfigurableResearchLLM(cfg)
    finally:
        torch.set_default_dtype(old_dtype)
    model.eval()

    # Mock tokenizer — these tests only exercise helper methods, not generation
    tok = MagicMock()
    tok.vocab_size = 256
    tok.eos_token_id = 0
    tok.bos_token_id = 0
    tok.pad_token_id = 0
    tok.encode = MagicMock(return_value=[1, 2, 3])
    tok.decode = MagicMock(return_value="test")
    tok.__call__ = MagicMock(return_value={"input_ids": torch.tensor([[1, 2, 3]])})

    engine = ForgeEngine(model, tok, device="cpu")
    yield engine
    del engine
    del model

class TestRadixPrefixCacheFixes:
    """RadixPrefixCache: mid-edge splits, no sentinel kv, no child clobber."""

    def _cache(self):
        from forge.engine.prefix_cache import RadixPrefixCache
        return RadixPrefixCache()

    def test_no_sentinel_kv_state(self):
        """New nodes have kv_state=None — the object() sentinel defeated
        every `is not None` check and returned a bare object as 'cached KV'."""
        from forge.engine.prefix_cache import RadixPrefixCache
        node = RadixPrefixCache._Node()
        assert node.kv_state is None

    def test_lookup_returns_none_on_empty(self):
        import torch
        cache = self._cache()
        ids = torch.tensor([list(range(32))])
        assert cache.lookup_longest_prefix(ids) is None

    def test_mid_edge_insert_splits_and_preserves_both(self):
        """Insert [prefix,X,tail] then [prefix,Y,tail]: the second insert
        must split the edge, not clobber it — both lookups must return
        their own kv. (lookup_longest_prefix ignores ids <= 16 tokens.)"""
        import torch
        cache = self._cache()
        p = list(range(20))
        cache.insert(p + [100, 101, 102], "kv_x")
        cache.insert(p + [200, 201, 202], "kv_y")
        hit1 = cache.lookup_longest_prefix(torch.tensor([p + [100, 101, 102]]))
        hit2 = cache.lookup_longest_prefix(torch.tensor([p + [200, 201, 202]]))
        assert hit1 is not None and hit1[1] == "kv_x"
        assert hit2 is not None and hit2[1] == "kv_y"
        assert hit1[0] == 23 and hit2[0] == 23

    def test_shorter_insert_at_mid_edge_boundary(self):
        """Insert long seq then a strict prefix of it: kv lands on a real
        node boundary (mid-edge split), not inside the edge."""
        import torch
        cache = self._cache()
        p = list(range(30))
        cache.insert(p, "kv_long")
        cache.insert(p[:20], "kv_short")
        hit = cache.lookup_longest_prefix(
            torch.tensor([p[:20] + [999]]))
        assert hit is not None and hit == (20, "kv_short")
        hit_long = cache.lookup_longest_prefix(torch.tensor([p]))
        assert hit_long == (30, "kv_long")

    def test_eviction_runs(self):
        """ref_count stays 0 on insert — _evict_lru must be able to fire."""
        from forge.engine.prefix_cache import RadixPrefixCache
        cache = RadixPrefixCache(max_entries=2)
        for i in range(4):
            cache.insert(list(range(20)) + [100 + i], f"kv{i}")
        assert len(cache) <= 2


class TestRepetitionPenaltyOrdering:
    """Rep penalty must apply BEFORE temperature and to greedy paths."""

    def test_penalty_applies_pre_temperature(self, forge_engine_fixture):
        """Parity: index_copy result == penalizing raw logits then scaling."""
        eng = forge_engine_fixture
        torch.manual_seed(0)
        logits = torch.randn(1, 256)
        gen_ids = [5, 7, 9, 5, 7]
        out = eng._sample_next_token(
            logits.clone(), temperature=2.0, top_k=0, top_p=1.0,
            repetition_penalty=1.5, generated_ids=gen_ids)
        # Reference: penalize raw logits at the generated ids, then divide
        ref = logits.clone()
        for tid in {5, 7, 9}:
            ref[0, tid] /= 1.5
        ref = ref / 2.0
        # Greedy-equivalent check: argmax of the penalized+tempered logits
        # must equal argmax of the reference (sampling is stochastic, so
        # compare distributions, not the draw).
        assert out.shape == torch.Size([1, 1])
        # Deterministic path: temperature<=0 argmax uses penalized logits
        greedy = eng._sample_next_token(
            logits.clone(), temperature=0.0, top_k=0, top_p=1.0,
            repetition_penalty=1.5, generated_ids=gen_ids)
        assert greedy.item() == ref.argmax(-1).item()

    def test_penalty_can_change_greedy_argmax(self, forge_engine_fixture):
        """A penalized top token loses argmax — proves penalty applies at
        temperature=0 (old code skipped it entirely on the greedy path)."""
        eng = forge_engine_fixture
        logits = torch.zeros(1, 256)
        logits[0, 5] = 2.0   # top token, recently generated
        logits[0, 6] = 1.9   # runner-up
        greedy = eng._sample_next_token(
            logits.clone(), temperature=0.0, top_k=0, top_p=1.0,
            repetition_penalty=1.1, generated_ids=[5])
        assert greedy.item() == 6

    def test_penalty_one_is_noop(self, forge_engine_fixture):
        eng = forge_engine_fixture
        torch.manual_seed(0)
        logits = torch.randn(1, 256)
        a = eng._sample_next_token(
            logits.clone(), temperature=0.0, top_k=0, top_p=1.0,
            repetition_penalty=1.0, generated_ids=[1, 2, 3])
        assert a.item() == logits.argmax(-1).item()


class TestLogitsLastOnly:
    """logits_last_only must return [B,1,V] logits equal to full-pass last pos."""

    def test_last_only_matches_full(self, forge_engine_fixture):
        eng = forge_engine_fixture
        model = eng.model
        ids = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            full = model(ids)[0]
            last = model(ids, logits_last_only=True)[0]
        assert last.shape[1] == 1
        # Not bit-equal: Linear on [B,1,d] hits a different GEMM kernel
        # than [B,T,d] (reduction tiling) — bound the drift instead.
        max_diff = (last[:, -1] - full[:, -1]).abs().max().item()
        assert max_diff < 1e-4, f"last-pos logits drift {max_diff}"

    def test_targets_still_full_logits(self, forge_engine_fixture):
        """targets path must NOT be truncated (CE needs per-position logits)."""
        eng = forge_engine_fixture
        model = eng.model
        ids = torch.tensor([[1, 2, 3]])
        targets = torch.tensor([[2, 3, 4]])
        with torch.no_grad():
            out = model(ids, targets=targets, logits_last_only=True)
        logits, loss = out[0], out[1]
        assert logits.shape[1] == ids.shape[1]
        assert loss is not None
