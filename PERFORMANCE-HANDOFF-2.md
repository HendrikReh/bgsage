# neongammon performance review handoff 2

**Baseline** · `0dee4f3` on `main`, 2026-11-19. The [first handoff](PERFORMANCE-HANDOFF.md) records the completed work. This handoff covers five follow-up findings.

## Status

| Section | State |
| --- | --- |
| 1. Bearoff configuration in the cache identity | done in [#41](https://github.com/HendrikReh/neongammon/pull/41) |
| 2. Decision-only N-ply cube path | done in [#42](https://github.com/HendrikReh/neongammon/pull/42) (`BgBotAnalyzer.cube_decision`) |
| 3. Model reuse in the single-position analyzer | done in [#43](https://github.com/HendrikReh/neongammon/pull/43) |
| 4. Thread budget of one, work per worker | done in [#44](https://github.com/HendrikReh/neongammon/pull/44) |
| 5. One scoring routine for single and batch checker play | done in [#45](https://github.com/HendrikReh/neongammon/pull/45) (`score_checker_position`) |

A suspected gap noted during section 1 turned out not to exist: `MultiPlyStrategy`'s default N-ply tree hands the engine `*base_`, but `set_bearoff_db` replaces `base_` itself with a `BearoffStrategy` over the database, so the database applies throughout the tree, not only at the root. Checked 2026-11-20 on a position whose roots are not bearoff positions but whose leaves are: 2-ply equities with `multipy_set_bearoff_db` equal those of an N-ply strategy built around an explicit `BearoffStrategy`, and differ from those without the database.

The line anchors below point at `0dee4f3`; the sections above have moved code since.

Fix the bearoff cache identity first. Model and worker reuse increase the number of evaluations that can encounter the same cache entries. Next, add the decision-only cube path for steady-state speed, or tackle model reuse first if analyzer startup is the main cost.

Local measurements used Stage 11, an Apple M4 Pro, and the Release build with Accelerate. They are spot checks. For before-and-after measurements, keep the positions, model, build settings, and thread count fixed, and alternate the call order. [Build instructions](CLAUDE.md#L1068) · [Stage 11 configuration](python/neongammon/weights.py#L123)

## 1. Include bearoff configuration in the evaluation cache identity

**Finding.** `BearoffStrategy::eval_identity()` forwards the base evaluator identity. The cubeful tree uses that identity in its cache key even though the wrapper can replace neural estimates with database probabilities. A cache hit can therefore return an evaluation from the wrong configuration. [Bearoff identity](cpp/include/bgbot/bearoff.h#L154) · [Cache fingerprint and lookup](cpp/src/cube_eval.cpp#L505) · [Native binding that adds the database wrapper](cpp/pybind/bindings.cpp#L5656)

| 2-ply centered money equity | Neural call | Database-backed call |
| --- | ---: | ---: |
| Each configuration evaluated first on a fresh Stage 11 strategy | +0.304917 | +0.108504 |
| Same strategy, neural call first | +0.304917 | +0.304917 |
| Same strategy, database-backed call first | +0.108504 | +0.108504 |

**Reproduce.** Build a 26-entry board with entries `4: 3`, `6: 5`, `19: -6`, and `22: -1`, and zeros elsewhere. Load the default bearoff database. In one thread, call `bgbot_cpp.cubeful_probs_and_equity_nply` at depth 2 with a centered cube, using the same base strategy and alternating `bearoff_db=None` with the loaded database. Repeat in the opposite order with a new base strategy.

**Fix.** Give the cache a stable identity for the effective database configuration. Calls with the same base and database should share entries, including equivalent nested wrappers. A different database, or a reload that can change its data, should get a distinct identity. Apply the same rule to the shared rollout cache keys derived from this fingerprint. [Database load](cpp/src/bearoff.cpp#L721) · [Shared cache lookup](cpp/src/cube_eval.cpp#L601)

**Verify.** Compare warm equity and all five probabilities with fresh-strategy references in both call orders. Add coverage for distinct database instances, repeated use of the same database, and nested wrappers over the same base. Rerun bearoff, root-routing, and threaded strategy tests before broadening reuse. [Bearoff tests](tests/test_bearoff.py) · [Root-routing coverage](tests/test_backgame_category.py#L447) · [Threaded strategy tests](tests/test_cube_strategy_threads.py)

## 2. Add a decision-only path for N-ply cube calls

**Finding.** The unified binding evaluates ND and DT in the cubeful tree, then runs a separate cubeless search for the probability fields. `simulate_game` reads the offer and take flags. A decision-only call can return the original decision and equities directly after the first pass. [Two searches in the binding](cpp/pybind/bindings.cpp#L5780) · [Simulation cube calls](python/neongammon/simulation.py#L160)

| Depth | Both native passes | Decision pass only | Separate probability pass |
| --- | ---: | ---: | ---: |
| 2-ply | 1.77 ms | 0.81 ms | 1.06 ms |
| 3-ply | 24.56 ms | 8.62 ms | 13.36 ms |

These are separate medians from a serial C++ probe on 20 fixed money contact positions generated by random legal play with Python seed 772. Each pair used the same Release library, Stage 11 weights, default bearoff database, and analyzer cube-life settings. The probe cleared the calling thread's cache before each arm. Decision equities and double/take flags matched on all 20 positions at both depths. The timings cover the native work.

**Fix.** Expose a decision-only result for N-ply callers that need the flags or ND, DT, and DP values. Preserve `cube_action` and its full statistics for callers that read the probabilities or roll details. Let simulation use the cheaper path for supported policies while keeping its existing `cube_action` interface for custom players.

**Verify.** Compare ND, DT, DP, optimal equity, and all decision flags against `cube_action` at 2–4 ply. Cover money, Jacoby and beavers, match and Crawford, cube caps, bearoff, and a root-pinned position. Compare seeded simulation records before and after. Benchmark both entry points on the same positions with warmed analyzers. [Simulation tests](tests/test_simulation.py) · [Cube detail parity tests](tests/test_2ply_details.py) · [Fixed cube benchmark positions](scripts/profile_cube_benchmark.py)

## 3. Extend model reuse to the single-position analyzer

**Finding.** A 3-ply analyzer loads Stage 11 for its 1-ply strategy, loads it again when creating `MultiPlyStrategy`, and loads it a third time when the cached cube strategy is first requested. The batch implementation already builds N-ply strategies around a loaded base. [Analyzer base and cube strategy](python/neongammon/analyzer.py#L186) · [N-ply analyzer construction](python/neongammon/analyzer.py#L416) · [Factory that accepts a loaded base](cpp/pybind/bindings.cpp#L5456) · [Batch ownership](python/neongammon/batch.py#L126)

| Startup probe | Observed time |
| --- | ---: |
| Construct one Stage 11 base, median of 15 warm-filesystem calls | 19.0 ms |
| Construct a 3-ply analyzer with the default database, median of 15 calls | 46.3 ms |
| First empty default batch with a cold database-loader cache, one call | 2.4 ms |

The empty batch case was a single runtime check. The database loader registered one miss even though the wrapper returned an empty list. All three batch wrappers resolve models before checking for rows. [Database loading and model resolution](python/neongammon/batch.py#L87) · [Empty pre-roll batch](python/neongammon/batch.py#L234) · [Empty post-move batch](python/neongammon/batch.py#L329) · [Empty checker batch](python/neongammon/batch.py#L516)

**Fix.** Use `create_multipy_from_base` and reuse the immutable model weights for checker and cube analysis. Preserve independent evaluator configuration where root routing can differ. After fixing cache identity, share the default database where its lifetime and ownership are clear. Keep batch argument validation and load resources only when there are positions to evaluate.

**Verify.** Count model and database loads at construction, first checker play, first cube action, and subsequent calls. For empty batches, assert zero model and database loads and keep the existing invalid-level and conflicting-argument errors. Compare whole results using Stage 11, a 5-NN model, a custom database path, and database-disabled mode. Include root-pinned and bearoff positions. Measure construction, first analysis, and steady-state calls separately. [Model loading tests](tests/test_batch_models.py) · [Cube strategy reuse tests](tests/test_cube_strategy_reuse.py)

## 4. Tune thread selection and honor an explicit budget of one

**Finding.** The 1-ply native batch path starts and joins workers for each call. Separately, the N-ply analyzer clamps `parallel_threads=1` to two. A 3-ply checker-play probe with an explicit budget of one observed a two-worker Python pool. The money benchmark requests one internal thread because it can already run decisions concurrently. [Batch worker loop](cpp/pybind/bindings.cpp#L4091) · [Analyzer thread selection](python/neongammon/analyzer.py#L420) · [Checker executor](python/neongammon/analyzer.py#L1134) · [Benchmark caller](scripts/benchmark_money.py#L1550)

| Warm 1-ply batch size | One thread | Auto |
| ---: | ---: | ---: |
| 1 | 8.9 µs | 9.5 µs |
| 4 | 59.9 µs | 118.0 µs |
| 16 | 275.4 µs | 340.0 µs |
| 32 | 510.6 µs | 444.0 µs |
| 256 | 3.84 ms | 2.87 ms |

The probe reused one `BatchModels`, checked that serial and auto produced equal results, and alternated the two modes over 50 timed calls per size. The board list came from random legal play with Python seed 1429. The crossover is specific to this sample and workload.

**Fix.** Run the explicit one-thread checker path inline. In auto mode, choose a minimum amount of work per worker for cheap evaluations, calibrated by API and depth. The native benchmark code already uses depth-sensitive worker limits. Consider persistent workers only after fixing cache identity and checking that nested search respects the caller's overall budget. [Existing minimum-work dispatch](cpp/src/benchmark.cpp#L100)

**Verify.** Add explicit one-worker coverage at 2- and 3-ply and check that the inline path avoids constructing a pool. Compare complete results across budgets of one, two, four, and auto. Measure small and large batches and an outer-parallel benchmark run. Capture elapsed time and peak active threads so nested worker creation is visible. [Analyzer thread tests](tests/test_analyzer_thread_limits.py) · [Batch parity tests](tests/test_batch_checker_play_parity.py)

## 5. Share one scoring implementation between single and batch checker play

**Finding.** Python and C++ separately implement the 2-ply candidate filter, the cubeless survivor search and promotions, cube-aware rescoring, and the final top-two promotions. Both paths also have their own full-width orchestration at deeper levels. This finding is based on source inspection. No speedup was measured. [Analyzer path](python/neongammon/analyzer.py#L946) · [Python promotion](python/neongammon/analyzer.py#L1177) · [Native path](cpp/pybind/bindings.cpp#L305) · [Native survivor selection](cpp/pybind/bindings.cpp#L432)

**Fix.** Extract a native per-position scoring routine and route both entry points through it after it supports the single-position contract, including forced boards, cube caps, and cube rules. Keep the public result types and use each caller's thread budget. Preserve the exact searched set at 2-ply and full-width search at deeper levels. The cubeless promotions at 2-ply help determine which candidates enter the cube-aware search.

**Verify.** Compare full ordered move lists with the baseline at 1–4 ply where applicable, including boards, both equities, probabilities, equity differences, and depth labels. Cover equal equities, top-two promotions, forced boards, blocked rolls, root-pinned snakes, bearoff, match, and capped cubes. Existing parity tests cover the overlapping public options. Add cases for single-position options as they enter the shared routine. Benchmark public single calls and batches once both use it. [Batch parity tests](tests/test_batch_checker_play_parity.py) · [2-ply filter tests](tests/test_cubeful_nply_filter.py) · [Full-width tests](tests/test_cubeful_full_width.py)

## Regression checks

The second review ran the command below on `0dee4f3`. **114 passed and one skipped.** The skip is an expected 3-ply filter case because that depth searches every move. The runtime findings above came from targeted probes; use the verification steps under each finding to cover the new behavior.

```sh
PYTHONPATH=build:python build/venv/bin/python -m pytest -q -rs \
  tests/test_cubeful_full_width.py tests/test_cubeful_nply_filter.py \
  tests/test_batch_checker_play_parity.py tests/test_batch_mixed_rules.py \
  tests/test_batch_models.py tests/test_contact_cube_life.py \
  tests/test_2ply_details.py tests/test_legacy_pos_cache.py \
  tests/test_move_generator.py
```
