
#### FLUX knn extractive recall lift (R69 — held-out QA retrieval)

- **Pipeline** (`_knn_votes_dev`, all config-gated, defaults off):
  two-stage retrieval — broad cosine pool (`knn_pool`) -> query-
  content coverage rerank (`knn_cover`/`knn_cover_w`, high-idf
  prompt tokens only) -> votes.  QA-format twins can be suppressed
  (`knn_drop_qa`) or exploited (`knn_twin`: rows inside trained
  "Q:..A:ans" lines adopt the tokens after the answer scaffold).
  Vote sources: successor, span (`knn_span`, per-row `knn_dedup`,
  `knn_span_idf`), and corpus-wide spreading-activation field
  (`knn_spread*`) — a token votes the field at its occurrences,
  `knn_spread_doc` switches the proximity-decayed conv for a flat
  doc-coverage maxpool weighted by each token's own idf.
- **Fingerprints**: `knn_window` bag-of-features over the last W
  ring tokens (+ `knn_idf` rare-token weighting, `knn_residual`
  prose-mean removal, `knn_episode` prompt-span-only query fp via
  `_prompt_start`/`_prompt_g` set in `ingest()`).  Fixed the raw
  collapse where all fps sat at cos~0.99 with everything.
- **Merged decode** (`generate()`): `lg += knn_lift * log p_votes`
  for the first `knn_lift_steps` tokens — recall merges INTO
  generation instead of replacing it.  Gates: deep-order cell
  existence (`knn_lift_deep` — seen questions keep verbatim recall),
  thin-regime (`knn_lift_thin`), model confidence (`knn_lift_conf`),
  and vote concentration (`knn_lift_topconf` — diffuse votes are
  noise and cost more than they gain).
- **Bug fix — prompt self-match**: the spread field's live mask
  included the just-ingested prompt, so the question's own rare
  tokens self-activated and drowned evidence clusters.  The query
  span `[e0, e]` is now excluded from both the activation source
  and the vote-read positions.
- **Answer-type masks** (`attach_vocab`, `knn_atype`): word-initial
  alpha piece filter, `knn_tokmin`/`knn_tokmax` unigram band.
- **Measured capability profile** (4K trivia + evidence, v8):
  seen_qa 0.86-0.96; paraphrase generalization 54-80% (content-
  driven transfer where knowledge exists); held_qa ~5% — bounded
  by corpus knowledge (answers absent from the 4K docs, not by
  the mechanism: field correctly localizes content-bearing docs;
  gold tokens simply never co-occur with question words there).
- Tests 158/158 green.  Commits `03d50c8`, `6d929d7`, `eda1cb5`+.

#### FLUX gen_live (R68 — the live generalizer "gen head")

- **gen_live** (`config.py` `gen_live` + `gen_live_topm` +
  `gen_live_min_tot`, all default off/`4`/`-1`): consolidate_gen made
  incremental — the user's "info -> weights -> gen head -> params"
  loop closed.  Every learned position whose order cells hold
  evidence (post-write tot > min_tot) transfers the cells' recalled
  successor DISTRIBUTION — top-M followers, share-weighted — into
  the gen channels at `gen_live*conf`
  (`conf = max_k tot/(tot+backoff_kappa)`): gcell per-follower cell
  writes, gsyn/gsyn_deep/dnet blended soft targets
  `sum(share*_Reff[t])`, `_Rl` per-follower row EMAs.  The head owns
  NO weights or params — `_genlive_write` (statemixin) routes mass
  between existing stores; recall cells remain the knowledge
  substrate.
- **vs gen_recall**: gen_recall floors the realized token's own
  write; gen_live writes what recall *knows about the context* —
  including followers never observed at the position — and keeps
  writing after cells converge (realized writes stall at s->0,
  conf stays ~1).
- **Post-write cells on all three learn paths**: eager
  `_learn_token` reads `_tables`/`_totals` after the order write;
  `_bookkeep_fused` re-probes post-step (`_fused_seq` batches one
  post-row probe — later-same-key followers leak in, the
  consolidate_gen-class approximation); `_ingest_bulk` reuses the
  reward fix-up's `slot2`/`found2` post-add_multi re-probe —
  pre-write probes meant first-pass bulk ingests transferred
  nothing (caught by test_gen_live_gpu).
- **Gating**: `_predict_only` skips it; thin/first-seen contexts
  self-gate (post-write mass ~s stays below min_tot).
- **Undo**: all writes ride existing journal ops (G/GS/gs/gd/dn/
  rL/rB) — `revert_since` restores `_gc`/`_gct`/`_Rl` bit-exact and
  `_S` to fp16-factor precision.
- **Self-play**: `--flux-gen-live` (implies `--flux-gen-channels`),
  `--flux-gen-consolidate` per-epoch batch complement.
- **Probe** `.tmp/exp_gen_live.py`: gen stores gain +14-38% mass
  per ingest; novel-frame NLL neutral (transfers are distribution-
  consistent — benefit is continuous consolidation, not hallucinated
  mass).  Tests `test_gen_live_*` x6, 158/158 green.

#### FLUX dnet (iterated delta-rule memory — "the online transformer")

- **dnet** (`config.py` `dnet` + 8 knobs): a stack of gated delta-
  rule layers — the linear-attention/delta-net answer to "a
  transformer that doesn't bog down learning and isn't black-box".
  Each layer's recurrent state S_l [topic_dim, dnet_dk] is the SAME
  delta-rule memory as gsyn's _S; the stack gives what every
  single-step channel lacks: ITERATED content routing.  Layer-0
  reads the shared gsyn key; layer-l's input is relu(o_{l-1}), so
  one layer's readout addresses the next layer's memory — the
  primitive attention gets from depth.  relu is load-bearing
  (breaks the odd symmetry f(-u)=-f(u) that caps linear layers).
- **Minimal mem**: no KV cache (O(1) state at any context length),
  no optimizer state (delta rule is momentum-free) — ~0.34MB total
  for the default 2-layer stack (states fp32 + int8 bipolar
  projections).  Params are seeded knobs, never trained — learning
  is inference, per FLUX philosophy.
- **Gates**: content-dependent write gate beta_l = sigmoid(w_b·u_l)
  · dnet_beta · surprise (GDN-style); scalar retention dnet_decay.
- **Depth proof**: antipodal-key task — 1 layer = 2/4 (linear
  ceiling), 2 layers = 3/4 (strictly exceeds it; partial convergence
  at 2400 writes).  Iteration + relu provably lifts the ceiling.
- **Parity with FLUX systems**: own lift/thin/temp knobs, own
  journal ("dn" entries + blk["dnet"], added to revert boundary
  set), own explain label + memory_report row + probe_dnet()
  (per-layer S_norm/key_norm/read_norm/gate + top tokens).
  Write paths: eager, fused bookkeep, bulk (exact sequential —
  layer l needs l-1's current state, so the WY chunk shortcut
  doesn't lift), teach, consolidate_gen.  Readout contract
  identical to gsyn (log_softmax(_Reff @ norm(o_L) * temp)).
- Tests: 9 new (opt-in, write dynamics, thin-gated lift, depth >
  linear ceiling, journal+revert exact, snapshot roundtrip,
  consolidate, NaN guards, GPU).  158/158 green.

#### FLUX gsyn_deep (nonlinear test-time-trained memory)

- **gsyn_deep** (`forge/model/flux/config.py` `gsyn_deep` + 9 knobs):
  a separate 2-layer MLP memory channel f_theta(k) -> expected
  _Reff successor fingerprint, trained ONLINE by surprise-scaled
  gradient steps - TTT-MLP / Titans-style "the memory is a learned
  function" done in FLUX clothing.  theta = (Wd1 [h,gsyn_dim],
  Sb1 [h], Wd2 [topic_dim,h], Sb2 [d]) + heavy-ball momentum
  buffers; ~115K params, <0.5MB fp32, seed-derived init, Wd2
  zero-init so the channel is silent until taught.  Explicit
  gradients (no autograd): err = o - v, relu-masked backprop through
  both layers, global-norm clip (gsyn_deep_clip), momentum, then
  theta <- decay*theta - lr*s*(mu*M + g).  Chunked bulk path
  (_gsd_chunk) batches LaCT-style: one surprise-weighted mean-gradient
  step per gsyn_deep_chunk positions.
- **Kept fully separate from linear gsyn** per the dissection
  mandate: own lift weight (gsyn_deep_w), own thin gate
  (gsyn_deep_thin), own temp, own journal ops, own explain channel
  label ("gsyn_deep"), own memory_report row, own probe.  Ablations:
  gsyn-only / deep-only / both / neither all work.
- **Expressivity proof** (test_gsyn_deep_nonlinear_vs_linear):
  antipodal keys +u/-u -> same target is impossible for a linear
  map f(-u) = -f(u); measured linear gsyn = 2/4 (chance) vs
  gsyn_deep = 4/4 on the controlled XOR-corner task.
- **Undo model**: gradient steps have no closed-form inverse, so
  whole-theta pre-state snapshots ("gd" entries + blk["gsd"] inside
  "bc" blocks; _gsd_session flush = one entry per ingest/consolidate
  span).  Fixed a latent boundary bug: "gd" wasn't in revert_since's
  boundary set so a rewind overshot into the PREVIOUS span's session
  snapshot - "gd" added to the boundary tuple (same class as "rB").
- **Wiring**: _learn_token (eager), _bookkeep_fused (fused device
  steps, reuses gsyn_k), _ingest_bulk (blk["gsd"] snapshot +
  _gsd_chunk), teach() (repeated writes like gsyn), consolidate_gen
  (recall-distilled soft targets via the v= path, shared with the
  gsyn collection loop).  Readout: _gsd_logits = log_softmax(
  _Reff @ normalize(f_theta(q)) * gsyn_deep_temp), lift =
  w * _gen_bias * (logp - log_uni) - same contract as gsyn.
  probe_gsyn_deep() returns key norm / hidden-relu stats / output
  norm / top tokens / param+momentum norms / theta drift from seed.
- Snapshot: "gsd" = 8-tuple (theta+momentum) CPU tensors; load
  restores bit-exact when config.gsyn_deep, silently reseeds when
  off (config-off ablation).  to_device covers all 8 tensors.
- Tests: 10 new in test_flux.py (defaults/opt-in, write dynamics,
  thin-gated lift, nonlinearity proof, journal+revert exact,
  snapshot roundtrip, consolidate writes, explain label, NaN
  guards, GPU path).  149/149 test_flux + test_train_flux green.

#### STRATA M1 + FLUX r_live (training-free generation)

- **STRATA M1** (orge/model/strata/model.py, StrataLM): CPU-eager
  reference of the packed-cell successor model on the M0 substrate
  (CellBlock/BlockStore/ParamSet/KnobSet/io.py).  Rolling
  suffix-hash keys over the 10 configured orders, channel registry
  (order o3..o64 + uni/backoff/recency/fatigue), Hedge geometric
  mixture with uni_escape floor and sleeping-experts abstention,
  surprise-gated writes scaled by max(1-p_mix, surprise_floor).
  ingest/teach/observe/learn paths, episode-gated keys via
  uto_episode+EOS, soft_reset, orward() returning
  (logits, None, None) (internal-state model, no KV), generate()
  with temp/top-p/top-k/support_thresh knobs, explain() provenance
  (kind, block, slot, key, tag per fired cell), udit()/
  writes_since(), tag-scoped 
evert_tag/
evert_since on the
  store write-clock arenas + per-block undo, .strata snapshot/load
  (cell slabs + uni/ring/count/hedge aux tensors).  25 tests in
  	ests/unit/test_strata_m1.py mirror the FluxLM learn/recall probes
  (instant teach, filler-surviving recall, write locality, revert
  restores prediction, uni escape floor, hedge anti-collapse).
- **FLUX r_live** (
_live=True default, 
_eta knob): the no-SFT
  substitute for graft_teacher/distill_gen.  New additive drift
  table _Rl [V, topic_dim] EMAs each learned token's feature row
  toward the normalized PRE-x context sketch _c/|_c|*sqrt(d) at rate
  
_eta*s (surprise-scaled) - co-occurrence embeddings written by
  the stream itself.  _R stays the frozen seeded bipolar matrix so
  every context-side EMA (_c, _c2 bind pairs, knn/csem
  fingerprints) keeps its bit-exact inverse under revert; the
  vocab-facing gen readouts use the materialized blend
  _Reff = _R + _Rl (gsyn logits readout AND delta-rule write
  targets).  Result: gsyn/gcell generate sensibly with zero
  trained parameters - seeded _R/_Wk/_Wq params + knobs carry
  the whole prediction system; mem_bias dial unchanged.  Measured:
  after 8x 'the cat sat...' ingest, gsyn top-8 under r_live = only
  corpus-plausible tokens (sp,t,a,e,c,n) vs random tail with it off.
- **r_live write sites**: _learn_token (eager), _bookkeep_fused
  (reconstructs pre-x _c by inverting the topic EMA -
  c_pre = (_c - (1-lam)*R[x])/lam), _ingest_bulk (_rl_bulk
  grouped residual add on cprev rows, deterministic _group_totals
  accumulation, blk['rl'] first-touch undo dict), 	each() (target
  rows bind to the context vector).  Undo: 'rL' per-token /
  'rB' session entries restore _Rl rows and refresh _Reff;
  'rB'/'rL' added to the revert boundary set (session-scoped restores
  may legally end a rewind - same class as 'bc').  Snapshot persists
  'Rl'; _Reff re-derives on load.  graft_teacher still works -
  graft rebases, r_live then refines.
- **FLUX recall->gen** (`gen_recall` knob + `consolidate_gen()`):
  the recall side now feeds the gen side directly.  Live path: when
  a context is *warm* (pre-write order-cell evidence tot > thin_tot)
  gen-channel writes (gcell/gsyn/_Rl) use `max(surprise, gen_recall)`
  as their strength, so recall-proven successors write through even
  when unsurprising - wired in all three write paths: _learn_token
  (pre-write probe hoisted ahead of the cell write), _bookkeep_fused
  (device-side `boff_thin` warm flag), _ingest_bulk (pre-write
  tots_f/found_f probes).  Batch path: `consolidate_gen(mass, top_m,
  min_tot, tag)` replays on-ring positions with closed-form pre-token
  contexts (_cprev_seq/_c2prev_seq seeded at zero inside the episode),
  reads each context's order-cell successor distribution across all
  orders, and writes the top-m share-weighted followers into gcell
  cells, the gsyn delta state (explicit target vectors via the new
  `v=` path in _gsyn_chunk), and _Rl rows - the gen side distills
  what recall already proved without re-ingesting.  All writes sit
  inside one ws/we journal block (gsyn "gs" state snapshot, "rB"
  session, per-cell "G") - revert_tag restores S/_Rl/gcell
  bit-exactly.  `_ringg` got a host fallback so _gcell_feats_bulk
  runs on CPU models; `_pows`/`_cprev_seq`/`_c2prev_seq` gained
  seed-at-zero args for episode-floor replay.
- **Measured**: 21.7K-token topic corpus -> 14.5s ingest (1501 tok/s
  CPU, all channels on) + 5.2s consolidate (26.5K positions, 48.9K
  gcell + 26.5K gsyn + 48.9K rl writes); held-out next-token acc
  64.3%; argmax continuation stays on-topic.  ~20s to basic topic
  competence vs Mini-AGI's ~2h for basic English - different
  mechanism (associative memory, not gradient learning) but the
  ingest+recall->gen loop is the fast path the user asked for.
- **Fixed while wiring**: `_gsyn_chunk` gam<=0 sequential fallback
  ignored the `v` param; "gs" op added to revert dispatch; STRATA 
evert_tag grabbed the LAST
  tagged token's position (reversed scan) so rewinds stopped at the
  boundary instead of the tag's first token - now min() over all
  matching entries.  FLUX snapshot roundtrip of drifted features:
  _R was only persisted under 'R_graft' (seed-derived assumption
  invalidated by r_live) - _Rl now its own key.
- Tests: 	est_strata_m1.py (25) + 6 r_live tests in
  	est_flux.py (drift locality, semantic convergence, revert row
  restore, no-teacher gsyn readout, flag-off parity, snapshot
  roundtrip).
- **R60e — fuzzy channels join hedge arbitration + candidate
  restriction** (measured fix for the ablation regression): gcell,
  gsyn, gsyn_deep (as "gsd") and dnet are now real hedge channels
  (appended at the end of `channels`, flag-gated) - real probs in
  `_channel_prob` (eager) and `real_r` under `bulk_channel_rewards`
  (bulk, post-block state approximation), abstain-neutral on the
  fused device step.  Lifts multiply by w_rel = C * w (relative to
  uniform share: 1.0 at init, sinks when losing) instead of the old
  fixed-weight bypass.  New knobs `lift_topk=64`/`lift_topm=8`:
  fuzzy lifts vote only inside top-k of the pre-lift row + own
  top-m proposals (rerankers, not mass injectors; 0 disables).
  `_sparse_lift_vec_dev` splits lift (dict cells, unrestricted) vs
  flift (gcell, restricted); `explain()` keeps raw channel output.
  Load: `w` name-remap already covers new names; `wR` shape-mismatch
  now replicates the global row.  Eval harness added:
  `.devin/eval_flux_quality.py` (seen/held-QA, held continuation,
  JSON tool-call parse) + `.devin/eval_flux_grid.py` (channel
  ablation driver, JSONL out).  Numbers @2K docs n=50 - pre-fix
  all-channels seen_qa 0.28 vs recall-only 0.82; post-fix all=0.84,
  unres(lift_topk=0)=0.84 (hedge alone carried it), bcr=0.84,
  sharp1=0.98.  Held-out still 0 at 2K docs - generalization is
  the remaining wall.
