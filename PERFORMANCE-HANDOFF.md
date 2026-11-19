# bgsage performance review handoff

Start with the **mixed-batch rule fix** (section 3): it is a few lines of Python and fixes wrong results. Then **skip the discarded cubeless search in cubeful checker play at 3-ply and deeper** (section 1), then make **batch 2-ply checker play follow the analyzer's filter** (section 4). Fix repeated model construction next for workloads that submit small batches. Keep changes to search policy separate from performance work so result differences are easy to attribute.

## Status

| Section | State |
| --- | --- |
| 1. Cubeless pre-pass at 3-ply+ | done in [#28](https://github.com/HendrikReh/bgsage/pull/28) |
| 2. Batch model construction | done in [#33](https://github.com/HendrikReh/bgsage/pull/33) (`BatchModels`) |
| 3. Mixed-batch rules | done in [#27](https://github.com/HendrikReh/bgsage/pull/27) |
| 4. Batch 2-ply filter parity | done in [#30](https://github.com/HendrikReh/bgsage/pull/30); see the notes below |
| 5. Details path | done in [#36](https://github.com/HendrikReh/bgsage/pull/36): the details run on the engine, so the headline equals the plain call exactly; results change (see the PR) |
| 6. Cache clearing | done in [#34](https://github.com/HendrikReh/bgsage/pull/34): the legacy table is allocated on first insert |
| 7. Move generator | done in [#35](https://github.com/HendrikReh/bgsage/pull/35) |

Notes on #30:
- It found three more differences between batch and analyzer, besides the filter:
  - the batch took its probs from the cubeless tree;
  - it routed the cube tree per node, not from the root;
  - it searched match positions at cube value 1.
- The batch now returns the analyzer's list field for field (bearoff database off).
- With mixed cube states, the best move had differed on 4 of 40 positions at 2-ply.
- The line anchors below point at `f164c92`; every section has moved code since.

## Baseline and current behavior

The original review and local spot checks used `6a3a269`, Stage 11, and the Apple Silicon Release build with Accelerate on an M4 Pro. The handoff was checked against `f164c92`, which includes `bef8bcd` and changes cubeful **2-ply** checker play to search a filtered set. [Apple Silicon build instructions](CLAUDE.md#L1035) · [2-ply filter benchmark](CLAUDE.md#L3163)

**Verification pass (2026-11-19, `f164c92`).** Every finding below was checked against the code and, where it makes a runtime claim, re-measured with the installed Release extension (load average ~3). All line anchors in this file point at the right code on `f164c92`. The re-measured numbers sit next to the historical ones; where they differ, the re-measured ones are the ones to plan with. The probes generated positions by random legal play from `STARTING_BOARD` (a random number of half-moves in [4, 40), keeping rolls with at least two legal moves; the seed is given per probe). Establish fresh before-and-after timings from the same starting commit, with fixed boards, model weights, thread counts, and build settings, before quoting a speed-up.

| Section | Code claim | Runtime claim | Correction |
| --- | --- | --- | --- |
| 1. Cubeless pre-pass at 3-ply+ | confirmed | identical results confirmed | saving is 12–24%, not 34% |
| 2. Batch model construction | confirmed | confirmed | absolute times ~20% lower |
| 3. Mixed-batch rules | confirmed | reproduced exactly | the fix is Python-only |
| 4. Batch 2-ply filter parity | new | 2 of 40 picks differ | missing from the original review |
| 5. Details path | confirmed | overhead ×1.3, not ×1.8 | headline shift up to 0.05, not 0.0016 |
| 6. Cache clearing | confirmed | not re-timed | <1% of a 3-ply call |
| 7. Move generator | confirmed | — | maintenance only |

## 1. Skip the discarded cubeless search in cubeful checker play

**Finding.** The cubeful wrapper calls the cubeless analyzer before deciding how to evaluate candidates. At full-width depths (3-ply and deeper, everything not in `CUBEFUL_FILTERED_PLIES`) it replaces every candidate's probabilities, equities, and depth label, then sorts the results again; nothing of the first pass survives except the candidate boards. The first pass generates and scores moves, searches survivors, promotes candidates, and clears caches. `_move_order_key` is a total order (cubeful equity, cubeless equity, board), so the input order cannot leak into the result either. [Cubeful wrapper](python/bgsage/analyzer.py#L942) · [Cubeless checker pass](python/bgsage/analyzer.py#L433) · [Cube-aware evaluation](python/bgsage/analyzer.py#L1056)

| Historical depth (`6a3a269`) | Existing call | Enumeration-only pre-pass | Median reduction |
| --- | ---: | ---: | ---: |
| 2-ply | 19.4 ms | 14.2 ms | 27% |
| 3-ply | 156.4 ms | 103.9 ms | 34% |

The historical experiment reused one analyzer per depth with two threads. It alternated the original and bypassed calls over 10 positions: three opening cases and positions taken before zero-indexed turns 7, 13, 19, 25, 35, 45, and 55 of random legal play with Python seed 11, with money, match, Crawford, and capped-cube cases. The 2-ply row predates the filter and cannot estimate savings on current 2-ply. The merged filter separately reported a 2.2–2.4× speedup and a 2.2–2.9 checker PR improvement on the snake folder. [Filter rollout and benchmark results](CHANGELOG.md#L57)

**Re-measured on `f164c92`** (3-ply, the bypass replaces the inner call by `possible_moves` enumeration only):

| Probe | Result |
| --- | --- |
| Parity, 10 positions (seed 11), cycling money / 5-away 3-away / Crawford 1-away 4-away / capped cube (2, player, max 2) / opponent-owned cube 2 | full ordered move list identical on 10 of 10 (boards, both equities, all probs, equity differences, depth labels) |
| Timing, 24 positions (seed 59), 2 threads, order alternated | median 125.7 → 104.0 ms (−17%); total 3.92 → 3.45 s (−12%); pre-pass alone 16.5 ms |
| Timing, same positions, 10 threads | median 40.6 → 30.9 ms (−24%); total 1.13 → 0.95 s (−16%); pre-pass alone 7.0 ms |

So the bypass is safe and worth doing, but plan with 12–24%, not 34%. The pre-pass's share grows with the thread count because the cube-aware pass parallelises across candidates and the pre-pass mostly does not.

**Implementation.** At 3-ply and deeper, generate the legal boards directly and send them through the existing cube-aware evaluator. At 2-ply, the preliminary pass determines which moves receive deeper evaluation. It currently includes candidates promoted by the cubeless search, and an existing test requires that superset (`test_best_move_comes_from_the_search`: every cubeless survivor is searched in the cubeful path too). Treat any change to which candidates enter the 2-ply search as a separate decision backed by paired accuracy measurements. [Filtered candidates and 1-ply fallback](python/bgsage/analyzer.py#L1100) · [Cubeful promotion](python/bgsage/analyzer.py#L1174) · [Survivor coverage test](tests/test_cubeful_nply_filter.py#L81)

The bypass also drops the pre-pass's `self._strategy_nply.clear_cache()` call. The cube-aware pass runs in a fresh `ThreadPoolExecutor` per call and the engine's cache keys carry a process-unique evaluator id, so this should not change results; the parity check above agrees, but keep it in the regression test.

**Verification.** Compare the full ordered move list before and after, including probabilities, both equities, equity differences, and depth labels. Cover contact, bearoff, a blocked roll, root-pinned snakes, forced boards, Crawford, and a capped cube (the probe above covered the cube cases, not bearoff, blocked rolls or snakes). Run the filter regression tests and checker/cube symmetry tests. Time checker play at 3- and 4-ply separately with the same warmed analyzers and position list. [Filter tests](tests/test_cubeful_nply_filter.py) · [Multi-ply symmetry test](tests/test_checker_cube_symmetry.py#L151) · [Root routing test](tests/test_backgame_category.py#L447)

## 2. Reuse loaded models across batch calls

**Finding.** Batch entry points construct strategies on each call. N-ply checker batches build both a 1-ply scorer and an N-ply strategy, whose factory (`create_multipy`) loads a second base model. Empty batches still pay setup costs. The C++ `MultiPlyStrategy` constructor already takes a base strategy, but the Python class exposes no constructor, so sharing the base needs a new binding or factory argument. [Pre-roll batch setup](python/bgsage/batch.py#L101) · [Post-move batch setup](python/bgsage/batch.py#L232) · [Checker batch setup](python/bgsage/batch.py#L453) · [N-ply factory](cpp/pybind/bindings.cpp#L5590)

| Stage 11 operation | Historical median (`6a3a269`) | Re-measured (`f164c92`) |
| --- | ---: | ---: |
| Construct a base strategy | 20.252 ms | 15.76 ms |
| Empty 1-ply pre-roll batch | 19.771 ms | 15.75 ms |
| One-position 1-ply pre-roll batch | 20.232 ms | 15.67 ms |
| Native one-position call through a reused strategy | 0.006 ms | 0.0044 ms |
| Empty 2-ply checker batch | 39.889 ms | 31.73 ms |

Historical medians used 25 timed calls after warm-up, the re-measured ones 15 (200 for the native call). The one-position probe used the starting board and one worker. The native call isolates evaluation and binding overhead. It uses native defaults and excludes construction and high-level Python result conversion. The absolute times differ with machine load; the conclusion does not: a small batch costs one or two weight loads.

**Implementation.** Validate the level before returning an empty result. Give repeated callers an explicit way to own and reuse loaded strategies. Construct the N-ply checker strategy from the base already loaded for the batch, using the existing constructor. Centralize model-family dispatch where possible. [Base-strategy constructor](cpp/src/multipy.cpp#L276) · [Existing analyzer strategy reuse](python/bgsage/analyzer.py#L195)

**Verification.** Count factory calls to show that empty batches load zero models, an N-ply checker batch loads one base, and repeated calls can reuse it. Compare 5-NN and Stage 11 results at 1-ply and N-ply, including threaded batches and custom weights. Time cold creation, warm calls, and batches of 0, 1, 32, and 256 positions separately. An invalid level on empty input already raises today (checked on `f164c92`); keep that.

## 3. Apply cube rules per position in batches

**Finding.** The Python wrappers turn Jacoby (and, in `batch_evaluate`, beavers) off for the whole batch if any row is a match. The C++ side does not need this: `CubeInfo::jacoby_active()` already requires `is_money()`, and the beaver is only applied on the money paths of `cube.cpp`. So the native batch functions already handle a mixed batch correctly row by row, and the Python fix is to pass `jacoby` / `beaver` through unchanged. The behaviour is known: `test_batch_one_ply_matches_the_analyzer` evaluates one position per call to avoid it. [Pre-roll rule selection](python/bgsage/batch.py#L120) · [Post-move rule selection](python/bgsage/batch.py#L250) · [Checker rule selection](python/bgsage/batch.py#L464) · [Workaround in the test](tests/test_contact_cube_life.py#L84)

**Reproduction.** With Stage 11, run each 1-ply batch API with a centered money position from `STARTING_BOARD` (roll 3–1 for checker play). Then append the same board at 5-away/3-away. With default Jacoby and one worker, all three APIs change the money row (reproduced exactly on `f164c92`):

| API | money row alone | same row in a mixed batch |
| --- | ---: | ---: |
| `batch_evaluate` (cubeful equity) | +0.113318 | +0.118125 |
| `batch_post_move_evaluate` (cubeful equity) | −0.113318 | −0.118125 |
| `batch_checker_play` (best move equity) | +0.235920 | +0.247203 |

**Implementation.** Pass `jacoby` and `beaver` straight to the native calls and drop the `has_match` logic. Then drop the one-call-per-position workaround from the test.

**Verification.** Add a small checked-in fixture that works without `contact.bm`. For money-only, match-only, and mixed batches, compare each row to the same row evaluated by itself with the same options. Cover both batch orders, a centered cube, and a money position where the beaver flag matters. Compare homogeneous-batch output to the starting revision (it must not change). [External-fixture skip](tests/test_batch.py#L119)

## 4. Make batch 2-ply checker play follow the analyzer's filter (new)

**Finding (not in the original review).** Both N-ply `batch_checker_play` overloads compute `compute_cubeful` — the full N-ply cube tree — for **every** legal candidate, survivors and pruned moves alike, and additionally run the cubeless N-ply `evaluate_probs` on the survivors. So batch checker play still has the pre-`bef8bcd` behaviour at 2-ply: pruned moves carry 2-ply cubeful equities, and the pick can land on a move the filter rejected. The analyzer comment, the `bef8bcd` commit message and the docstring of `tests/test_cubeful_nply_filter.py` all say pruned moves keep 1-ply values "as in batch_checker_play"; that is not what the code does. At 3-ply and deeper both paths score every move through the cube tree, so they agree there. [Pruned candidates in the cube tree](cpp/pybind/bindings.cpp#L4726) · [Same in the second overload](cpp/pybind/bindings.cpp#L4990) · [Analyzer comment](python/bgsage/analyzer.py#L1100)

**Measured on `f164c92`.** 40 positions (seed 7), centered money cube, `batch_checker_play(..., eval_level="2ply", n_threads=4)` against `BgBotAnalyzer(eval_level="2ply", parallel_threads=2).checker_play`: the best move differs on **2 of 40**, and on moves the analyzer prunes the two equities differ by up to **0.43** (1-ply Janowski against the 2-ply cube tree). `test_batch_checker_play_matches_the_analyzer[2ply]` passes only because its single position (`CONTACT`, 6-2) happens to agree.

**The promotion loop recomputes cubeful equity.** The point of the original review still holds, as the smaller part of this: the stale-top-two loop calls `compute_cubeful` again for a move whose cubeful equity was already computed by the same deterministic function; only the N-ply probs are new. [First promotion loop](cpp/pybind/bindings.cpp#L4756) · [Second promotion loop](cpp/pybind/bindings.cpp#L5019)

**Implementation.** Consolidate the two N-ply overloads (they differ only in the `GamePlanStrategy&` / `Strategy&` parameter) around one scoring routine. In it, at the depths in `CUBEFUL_FILTERED_PLIES`, run the cube tree only for the filter's survivors and give pruned moves their 1-ply Janowski value (already computed in the 1-ply scoring pass); on promotion, run the cube tree and the N-ply probs once. This cuts the cube-tree calls at 2-ply from all candidates to the survivors, which is where most of a batch's time goes, and restores parity with the analyzer. It changes 2-ply batch results, so ship it as its own PR, separate from the pure performance changes; the accuracy case was already made for the analyzer in `bef8bcd`. Pass the filtered depth set from Python rather than hard-coding a second copy in C++. Fix the analyzer comment and the test docstring in the same PR.

**Verification.** Extend `test_batch_checker_play_matches_the_analyzer` to a few dozen positions at 2-ply and 3-ply, comparing the full ordered move list (boards, both equities, depth labels), not only the top move. Use a position that forces promotion to check that the cube tree runs once per promoted candidate. Time 2-ply batches of 32 and 256 positions before and after.

## 5. Share fast candidate scoring with detailed cube analysis

**Finding.** `incl_2ply_details=True` uses a separate recursion. Its move picker uses the grouped `batch_evaluate_candidates_best_prob` only for a `GamePlanStrategy` (5-NN models); for Stage 11 it falls back to a scalar loop over candidates. The plain path uses grouped NN inference. The two paths also select moves under different equity policies (cubeless 1-ply in the details path, cubeful 1-ply against the primary cube state in the plain path). [Detailed recursion](cpp/src/cube.cpp#L1982) · [Detailed move picker](cpp/src/cube.cpp#L1783) · [Grouped inference](cpp/src/cube_eval.cpp#L249) · [Plain move-selection policy](cpp/src/cube_eval.cpp#L737)

| Depth | Historical plain / details (`6a3a269`) | Re-measured plain / details (`f164c92`) |
| --- | ---: | ---: |
| 2-ply | 4.139 / 7.771 ms (×1.88) | 1.09 / 1.43 ms (×1.35) |
| 3-ply | 18.485 / 33.148 ms (×1.79) | 8.52 / 10.73 ms (×1.31) |

The historical spot check used nine fixed money positions, with one warmed analyzer per depth and two threads: the starting board and positions after every seventh turn from zero-indexed turn 0 through 49 of random legal play with Python seed 59. The re-measurement used nine positions from the random-play generator above (seed 59), one warmed analyzer per depth, two threads. The plain path is now much faster than at `6a3a269` (the cube strategy is loaded once per analyzer since PR #3), and the details overhead is about ×1.3, not ×1.8.

**The headline shift is larger than recorded.** Historically, changing the details flag moved headline ND or DT equity by up to 0.0016 on two boards. Re-measured at 2-ply on the nine positions: up to **0.007** on ND/DT in ordinary positions, and **0.0496** on DT (−3.5096 plain, −3.4600 with details) in a position that is a clear pass. Emptying `MONEY_CONTACT_CUBE_LIFE` gives the same shift, so the cube life is not the cause; the different move-selection policy is the likely one. This contradicts `CLAUDE.md`'s statement that at 2-ply the details call's "headline equities match the plain (no-details) 2-ply call". The optimal actions were not compared.

**Implementation.** Share candidate grouping and compatible leaf evaluation first (a grouped best-cubeless-pick for any `Strategy`, or route through `eval_groups_pair`). Keep the detailed path's cubeless move-selection policy explicit, including candidate order and tie handling. Replacing that policy with plain cube-aware selection would change the per-roll boards and headline values. Decide separately whether the headline shift is acceptable or whether the details call should report the plain call's headline equities, and correct `CLAUDE.md` either way.

**Verification.** Capture the detailed results from the starting revision. Compare ordered rolls, chosen boards, ND/DT equities, weighted totals, and opponent-double/pass handling on money and match positions. At 2-ply, check that the opponent-roll level is absent. Compare grouped and scalar numerical drift and inspect near-tie candidates. Time details and plain analysis from 2- through 4-ply. [Details tests and tolerances](tests/test_2ply_details.py) · [Fixed cube benchmark positions](scripts/profile_cube_benchmark.py)

## 6. Narrow cache clearing

**Finding.** `MultiPlyStrategy::clear_cache()` clears both thread-local tables: the engine's cubeful cache and the legacy `PosCache`, which the default tree bypasses. The legacy table holds 262,144 entries of 136 bytes (hash, plies, 26-int board, 5 probs), about **36 MB per thread**; every clear memsets all of it, and the first call on a thread allocates it. The historical warm clear took about 0.21 ms over 100 calls (not re-timed). Against a 3-ply checker call of 30–125 ms that is under 1%; the memory per calling thread may matter more than the time. [Cache clearing](cpp/src/multipy.cpp#L343) · [Default evaluator dispatch](cpp/src/multipy.cpp#L477) · [Legacy table size and zeroing](cpp/include/bgbot/multipy.h#L391)

**Implementation.** Low priority. Clear only the caches used by the relevant evaluator during routine analysis, or allocate the legacy table lazily on first insert. Preserve an explicit cold-cache operation for profiling and comparisons, and keep the engine cache clear (`CLAUDE.md` documents "call `clear_cache()` between strategy comparisons"). Before broadening reuse, check that the key identifies the active move prefilter and its effective depth policy. [Engine cache lookup](cpp/src/cube_eval.cpp#L477) · [Depth-sensitive pruning](cpp/src/cube_eval.cpp#L621)

**Verification.** Compare fresh and reused results across model and cube contexts, distinct prefilters, root pins, and parallel calls. Exercise both the default and legacy evaluator. Measure first-call cost, repeated related positions, and unrelated positions. Record cache hits and process memory alongside elapsed time. [Strategy reuse tests](tests/test_cube_strategy_reuse.py) · [Threaded strategy tests](tests/test_cube_strategy_threads.py) · [Cold-cache benchmark](scripts/bench_multipy_reference.py)

## 7. Put dice-use rules in one move generator

**Finding.** The two sorted overloads duplicate the same traversal word for word; they differ only in a local versus `thread_local` scratch buffer. The unsorted path repeats the legality rules with a faster insertion-time deduplication scheme. This is a maintenance opportunity with no projected speedup. [Returning overload](cpp/src/moves.cpp#L173) · [Buffer overload](cpp/src/moves.cpp#L216) · [Unsorted generator](cpp/src/moves.cpp#L342) · [Ordering contract](cpp/include/bgbot/moves.h#L26)

**Implementation.** First make the returning overload delegate to the buffer overload. Then benchmark sharing the legality traversal and deduplication. Preserve the sorted result order and the unsorted path's first-seen order where callers use it to break ties.

**Verification.** Compare the old and new outputs across all 21 rolls on fixed boards. Cover blocked bar entries, hits, partial doubles, the higher-die rule, overshoot bearoffs, and the stand-pat result. Check ordered candidates and roll aliases, then compare selected moves and checker benchmark PR. Time move generation separately on doubles-heavy positions and end-to-end checker play. The move-array tests compare APIs that can share the generator, so keep a before-build or a frozen corpus for independent parity. [Move-array tests](tests/test_move_arrays.py) · [Move-array benchmark](scripts/benchmark_move_arrays.py)

## Side observations (not checked further)

- The batch entry points never attach the bearoff database, while `BgBotAnalyzer` loads it by default (`bearoff_db=True`). Batch and analyzer can therefore disagree on bearoff positions.
- `batch_evaluate` and `batch_post_move_evaluate` pass `contact_x` only at 1-ply; at 2-ply they keep the 0.68 contact cube life. `CLAUDE.md` lists them under "at the matching levels". These batch paths use Janowski on N-ply cubeless probs rather than the cube tree, so whether the 2-ply value should apply there is a separate question.

## Test commands

Run from the repository root with the relevant Release extension in `build/`. Rebuild after native changes using the repository instructions. [Build setup](CLAUDE.md#L1035)

```sh
export PYTHONPATH=build:python
build/venv/bin/python -m pytest -q tests/test_cubeful_nply_filter.py \
  tests/test_checker_cube_symmetry.py::test_multi_ply_best_move_symmetry \
  tests/test_backgame_category.py tests/test_bearoff.py
build/venv/bin/python -m pytest -q tests/test_contact_cube_life.py \
  tests/test_batch.py tests/test_batch_checker_play.py -rs
build/venv/bin/python -m pytest -q tests/test_2ply_details.py \
  tests/test_cube_strategy_reuse.py tests/test_cube_strategy_threads.py
build/venv/bin/python -m pytest -q tests/test_move_arrays.py
```

On `f164c92` (2026-11-19), all four commands together give **85 passed and 9 skipped** in 16 s with the installed Release extension. The nine skips are `tests/test_batch_checker_play.py`, which needs `data/contact.bm`; that file is not in this checkout. The original review on `6a3a269` ran a subset (batch, checker-batch, strategy-reuse, and checker/cube symmetry files): 14 passed and 9 skipped. The latency, parity and mixed-rule evidence came from direct runtime probes, not from the test suite.
