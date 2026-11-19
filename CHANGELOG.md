# Changelog

Changes in this fork ([HendrikReh/bgsage](https://github.com/HendrikReh/bgsage))
on top of upstream [markbgsage/bgsage](https://github.com/markbgsage/bgsage).
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

The fork branched from upstream at `139427d` (2026-10-17, "Add compact
all-roll legal move arrays for video reconstruction"). Everything before that
commit is upstream's history; see `git log 139427d`. The fork targets macOS
on Apple Silicon only (see "Platform Scope" in `CLAUDE.md`). It has no
Releases keep upstream's `2.0.YYYYMMDD` version scheme, dated by the fork's
release. They are published as GitHub releases with Apple Silicon wheels,
not on PyPI, where the `bgsage` name is upstream's.

Timings below were measured on an M4 Pro Mac mini (10P+4E cores).

## [Unreleased]

### Added

- **Head-to-head games against XG** (`scripts/xg_h2h.py`). XG has no API,
  so each game is replayed to a fixed point instead:
  - Sage plays its seat and predicts XG's.
  - Each round writes XG's unanswered decisions as one-decision `.xg`
    games for one Batch Analyze.
  - The next round reads XG's picks back and re-plays everything after a
    disagreement.
  - A game is finished when every XG decision in it is XG's own.

  Dice are fixed per (game pair, turn), so a finished game is exactly what
  live play would have produced; `tests/test_xg_h2h.py` checks this with a
  Sage level standing in for XG. The opening is standard, each dice
  sequence is played twice with the seats swapped, and no game is ever
  skipped. `report --luck` gives luck-adjusted results; in the stand-in
  test that cut the per-game standard error from 0.89 to 0.11 points. The
  retired upstream replay runners stay retired; `CLAUDE.md` no longer
  describes them as working.

### Changed

- **The 1-ply money contact cube life now rises with the average pip
  count**: x = 0.7375 + 0.00175 x (average pips - 120), about 0.66 at 75
  pips and 0.79 at 150 (`MONEY_CONTACT_CUBE_LIFE_PIP_SLOPE` in
  `analyzer.py`).
  - Fitted to XG World Class's cube decisions on the money benchmark's 2,196
    cube positions (`scripts/fit_contact_cube_life.py`). Fitted on half the
    seeds and scored on the other half, the 1-ply cube PR falls by
    0.34 +- 0.11 against XG and 0.45 +- 0.11 against the Sage reference,
    compared with a flat cube life fitted the same way.
  - Undoes most of the blunder increase of the flat 0.75: 1-ply cube
    blunders against XG 42 -> 33.
  - Money benchmark at 1-ply: cube PR 2.36 -> 1.89, checker play unchanged.
    2-ply and deeper, races, match play and rollouts are untouched.
  - Engine: new `CubeInfo.contact_x_pip_slope` and `contact_cube_x()`; every
    binding that takes `contact_x` takes `contact_x_pip_slope`. C++ callers
    that don't pass it are bit-identical.

- **Money cube analysis at 1-ply and 2-ply uses a higher contact cube life**
  (0.75 and 0.80 instead of 0.68; `MONEY_CONTACT_CUBE_LIFE` in
  `analyzer.py`).
  - Both levels used to over-double, because their cube trees end in a 1-ply
    Janowski leaf.
  - The cube PR on the money benchmark falls from 3.21 to 2.36 at 1-ply and
    from 2.92 to 1.31 at 2-ply.
  - The value applies to every cubeful equity at that depth, so checker play
    and cube action stay consistent. Checker-play PR is unchanged within
    noise.
  - Races, match play, 3-ply and deeper, and rollouts are untouched. C++
    callers keep the old values unless they pass the new
    `CubeInfo.contact_x`.
  - Checked against XG World Class on all 2,196 benchmark cube positions
    (`scripts/export_money_cube_xg.py`): against XG the cube PR falls from
    2.74 to 2.00 at 1-ply and from 2.30 to 1.02 at 2-ply.
  - See "Cube life by depth (money)" in `CLAUDE.md`.

### Fixed

- **The test suite runs clean on macOS**
  ([#16](https://github.com/HendrikReh/bgsage/pull/16)): 318 passed,
  21 skipped, none failing. Three test files had upstream's Windows
  heritage, and none of the fixes touch engine code:
  - `test_bearoff.py` called the Windows-only `os.add_dll_directory` at
    import; the call is now limited to Windows.
  - `test_bearoff.py`, `test_rollout_progress_callback.py` and
    `test_gnubg.py` looked for `build/` (and `data/`, `models/`) in the host
    project; they now use this repo's.
  - `test_gnubg.py` skips when no GNUbg binary is found, and its top-100
    comparison skips when the gitignored `contact.bm` / `crashed.bm` are
    missing.

### Documentation

- `CLAUDE.md` records a paired comparison of the race-net swap against XG
  4-ply on all 5,117 decisions of the Benchmark PR sample. Stage 5's 200h
  net is never worse than the S6 100h net it replaced, but no gap is
  significant: overall PR −0.010 ± 0.008 at 1-ply and −0.038 ± 0.028 at
  2-ply.
  - At 2-ply the race gain is a single decision. The other 607 races
    outside the bearoff database net to −0.003.
  - At 1-ply the race gain is spread over many decisions: −0.088 ± 0.070.
  - At 2-ply the race net also changes 37 contact picks, for a gain within
    noise (−0.015 ± 0.007).
- `CLAUDE.md` records that re-rolling the containment net's general rows
  under Stage 9 did not help
  ([#17](https://github.com/HendrikReh/bgsage/pull/17)). All 291,242 rows
  were rolled out on Modal (~$223), and the retrained net's holdout ER fell
  (general rows 36.8 → 24.4), but its decisions did not improve. On the money
  benchmark's 237 containment-routed decisions it scores 3.50 / 2.05 / 1.28
  at 1P / 2P / 3P against 3.06 / 1.94 / 1.18 for the installed net, and the
  containment folder is a tie. The installed net stays.
- The `scripts/accuracy_vs_time.py` results were re-measured after the
  race-net swap ([#8](https://github.com/HendrikReh/bgsage/pull/8)).
  `stage11`'s Benchmark PR at 1 / 2 / 3 / 4-ply is now
  4.10 / 3.15 / 2.54 / 2.54, against 4.11 / 3.18 / 2.53 / 2.55 before.
  - Between 89 and 143 of the 5,140 picks changed per level. GNUbg's PR is
    identical to the first run.
  - Paired against GNUbg at the same depth: −0.17 ± 0.14, −0.29 ± 0.09,
    −0.21 ± 0.07 and −0.02 ± 0.08.
  - Open Sage 3-ply still matches GNUbg 4-ply (−0.02 ± 0.06) at about a
    fifth of the CPU time.

## [2.0.20261118] - 2026-11-18

The fork's first release.

### Added

- **Apple MLX trainer for supervised learning** (`bgsage.sl_train`,
  [#2](https://github.com/HendrikReh/bgsage/pull/2)). `supervised_train` and
  `supervised_train_preencoded` take the same arguments and return the same
  dict as the CUDA bindings. They use CUDA when available and otherwise Apple
  MLX on the Mac GPU (`pip install mlx`). All 20 training scripts now call
  them. The MLX path transcribes `cuda_nn.cu`. It is not bit-identical to
  CUDA, but `tests/test_sl_train.py` holds it to a float64 NumPy
  transcription within 2e-5. On the snake-net recipe it reaches holdout ER
  53.96, against 53.95 for CUDA on an RTX 4070S.
- **Configurable GNUbg binary** (`bgsage.settings`,
  [#1](https://github.com/HendrikReh/bgsage/pull/1)). The path is resolved
  from, in order: `set_gnubg_path()` or a per-instance `gnubg_path=`, the
  `BGSAGE_GNUBG` environment variable, `~/.config/bgsage/settings.json`, and
  auto-detection on `PATH`. `GnuBgAnalyzer` and `post_move_analytics` now run
  on macOS.
- **Cloud rollouts on Modal** (`scripts/modal_rollout.py`,
  [#7](https://github.com/HendrikReh/bgsage/pull/7)). The script rolls out
  position files on Modal CPU containers in the SL trainers' format, with a
  `.meta.jsonl` sidecar. Runs are resumable, and `--max-usd`, `--chunk` and
  `--local` are available. Measured cost is about $0.002 per containment
  board and $0.0001 per pure-race board.
- **Pure-race position harvest** (`scripts/harvest_race_positions.py`,
  [#7](https://github.com/HendrikReh/bgsage/pull/7)). 30,000 self-play games
  yield 329,753 train and 36,478 holdout race boards, as the basis for
  retraining the PureRace net on rollout targets.
- **Benchmark PR sampling** (`generate_benchmark_pr.py --resume
  --decision-step N`, [#1](https://github.com/HendrikReh/bgsage/pull/1)).
  This rolls out every Nth saved decision. The fork's reference is the
  1-in-20 sample: 6,944 decisions and 20,302 positions, about 4.5 hours.
  Both PR scorers accept `--decision-step N` and score exactly that sample.
- **CHANGELOG** for the fork ([#10](https://github.com/HendrikReh/bgsage/pull/10)).
- **`score_benchmark_pr_strategy` binding**
  ([#2](https://github.com/HendrikReh/bgsage/pull/2)).
  `score_benchmark_pr.py` now scores every registered model type: 5-NN,
  pair, and backgame-aware models including `stage11`.
- **Accuracy against thinking time** (`scripts/accuracy_vs_time.py`,
  [#10](https://github.com/HendrikReh/bgsage/pull/10)). The
  script measures Benchmark PR and single-thread CPU time per decision for
  Open Sage and GNUbg at 1–4 ply, with both engines picking moves as in
  normal use and depths in the XG convention. First results (`stage11`,
  5,140 decisions), PR at roughly equal CPU time: 2-ply 3.18 vs 3.44, 3-ply
  2.53 vs 2.75, 4-ply 2.55 vs 2.56. Open Sage 3-ply (~30 ms) matches GNUbg
  4-ply (~145 ms). The reference is Stage 5's rollout, which leans towards
  Open Sage.
- **XG as a Benchmark PR referee** (`scripts/export_benchmark_pr_xg.py`,
  [#8](https://github.com/HendrikReh/bgsage/pull/8)). The script exports the
  sample as one-decision `.xg` games for XG's Batch Analyze, reads XG's
  ranked moves back, and scores Sage and GNUbg against both the Stage 5
  rollout and XG. Against XG 4-ply, `stage11` 2-ply (before the race-net
  swap below) scores 1.24 and GNUbg
  2-ply 1.40 (paired difference −0.17 ± 0.07). The PR also adds
  `scripts/run_purerace_sl.py`, the pure-race fine-tuning experiment, which
  did not beat Stage 5's net.
- **Analysis tools** ([#6](https://github.com/HendrikReh/bgsage/pull/6)):
  - `scripts/race_gap_analysis.py` breaks down pure-race PR into bearoff
    positions and races outside the database.
  - `scripts/race_net_swap.py` scores the same races with other race nets
    swapped in.
  - `scripts/bench_analyzer_overhead.py` measures per-call latency of
    `cube_action` and `checker_play`.
- **Tests:**
  - `tests/test_encoding_golden.py` pins the encoder features on 1,000
    positions ([#1](https://github.com/HendrikReh/bgsage/pull/1)).
  - `tests/test_settings.py` covers the GNUbg path precedence
    ([#1](https://github.com/HendrikReh/bgsage/pull/1)).
  - `tests/test_analyzer_thread_limits.py` pins thread limits per level
    ([#3](https://github.com/HendrikReh/bgsage/pull/3)).
  - `tests/test_cube_strategy_threads.py` calls one analyzer from 8 threads
    and requires results bit-identical to a fresh analyzer per call
    ([#6](https://github.com/HendrikReh/bgsage/pull/6)).

### Changed

- **`stage11` uses Stage 5's 200h pure-race net**
  ([#8](https://github.com/HendrikReh/bgsage/pull/8)) instead of the 100h net
  every model since Stage 6 shared. XG as referee agrees with the swap in
  direction, though not significantly (see the paired comparison under
  [Unreleased]). Sample
  Benchmark PR at 1-ply / 2-ply: 4.11 / 3.17 → 4.10 / 3.14, all of it in the
  pure race (0.55 / 0.57 → 0.54 / 0.44). `stage11`'s pure-race results in
  `accuracy_vs_time.py` move slightly.
- **Release builds are Apple Silicon only.** `wheels.yml` builds macOS arm64
  wheels and attaches them, with the sdist, to the GitHub release of a `v*`
  tag. The PyPI job runs only in upstream's repository.
- **Apple Accelerate path for batched NN evaluation**
  ([#1](https://github.com/HendrikReh/bgsage/pull/1)). All batched candidate
  kernels go through `NeuralNetwork::forward_group`. On Apple arm64 it runs
  one `cblas_sgemm` per layer on the CPU's matrix unit (SME/AMX). It uses the
  exact Cephes sigmoid instead of the clamped NEON lookup table, so equities
  move by up to about 0.004. The path is on by default;
  `-DBGBOT_USE_ACCELERATE=OFF` restores the portable kernels, which are
  bit-identical to the previous engine.
- **Faster contact encoder** ([#1](https://github.com/HendrikReh/bgsage/pull/1)).
  `hitting_shots`, `double_hitting_shots`, `compute_piploss`, `moment2` and
  `backbone` use bitmask formulations. That takes 1.80 → 1.17 µs per board,
  bit-identical on 655k boards. Together with Accelerate, `checker_play` is
  1.37–1.41x faster at 2- and 3-ply. Money benchmark PR is unchanged (2-ply
  1.67 → 1.68, 3-ply 0.60).
- **Analyzer call overhead** ([#3](https://github.com/HendrikReh/bgsage/pull/3)).
  `cube_action` loads its base strategy once per analyzer instead of
  reloading all weight files (24 for `stage11`) on every call. The 1-ply
  cubeful `checker_play` conversion runs serially instead of in a thread pool
  (`SERIAL_CUBEFUL_1PLY`). Median speed-up per call: `cube_action` 1-ply
  about 1,500x, 2-ply about 13x, 3-ply about 2.3x, and `checker_play` 1-ply
  about 8x. Code that builds a new analyzer per call still pays the weight
  load.
- **`score_benchmark_pr.py` uses the bearoff database**, as the engine and
  GNUbg do ([#5](https://github.com/HendrikReh/bgsage/pull/5)). Only the
  pure-race column moves: `stage11` 1-ply 4.15 → 4.11, 2-ply 3.22 → 3.17.
  `--no-bearoff` scores the raw networks.
- **N-ply PR scoring adds the PubEval opponent prefilter** that
  `BgBotAnalyzer` uses ([#2](https://github.com/HendrikReh/bgsage/pull/2)).
  Stage 5 2-ply goes from 4.104 to 4.097.
- **Repo-root paths** ([#1](https://github.com/HendrikReh/bgsage/pull/1)).
  `generate_benchmark_pr.py`, `score_benchmark_pr_gnubg.py` and
  `test_match_gnubg_comparison.py` resolve `python/`, `build/`, `data/` and
  `logs/` inside this repo instead of the host project above it.
- **`test_match_gnubg_comparison.py` evaluates through `BgBotAnalyzer`**
  ([#1](https://github.com/HendrikReh/bgsage/pull/1)), so it runs with any
  model, not only 5-NN ones.

### Fixed

- **Post-Crawford cube for the trailer**
  ([#9](https://github.com/HendrikReh/bgsage/pull/9)). After the Crawford
  game the trailer doubles at the first opportunity, but `cl2cf_match`
  still interpolated towards a dead cube at the current value. On 15 BG
  Tutor duel positions (3- or 4-away vs 1-away, cube centered) this put
  2-ply No Double 0.058–0.184 equity below Double/Take, so analyses charged
  the trailer an error for not doubling. GNUbg 2- and 4-ply and rollouts
  from both engines put the two equal. The position is now valued as
  doubled once, as long as the trailer can still turn the cube. Every
  caller shares this one conversion: 1-ply, the N-ply leaves, rollout
  truncation, bearoff and the Python `cl2cf` binding. The Crawford game,
  leader-owned cubes, match-dead cubes and pre-Crawford scores are
  unchanged. `tests/test_post_crawford_cube.py` adds 24 cases; 16 of them
  fail on the code before the fix.
- **Thread limits for 1-ply cubeful checker play**
  ([#3](https://github.com/HendrikReh/bgsage/pull/3)). `parallel_threads=1`
  was replaced by the CPU count (`workers <= 1` instead of `workers <= 0`),
  and `_OnePlyAnalyzer` never received the limit.
- **GNUbg comparison without a cube decision**
  ([#1](https://github.com/HendrikReh/bgsage/pull/1)). At double match point,
  in the Crawford game and for the post-Crawford leader, GNUbg prints no cube
  analysis. These cases now count as agreement when bgsage also has no cube
  action, which takes the tally from 13 / 0 / 5 skipped to 18 agree.
- **Benchmark PR scoring counts only the rolled-out sample**
  ([#2](https://github.com/HendrikReh/bgsage/pull/2)). The scorers used to
  include decisions whose candidates happened to be covered, a skewed set
  (Stage 5 2-ply 3.37 on the sample against 5.27 on those extras).

### Documentation

- The fork is declared Apple Silicon only in `README.md` and `CLAUDE.md`,
  with a tested Apple Silicon build section. Windows, Linux and CUDA builds
  are kept but not supported
  ([#1](https://github.com/HendrikReh/bgsage/pull/1)).
- `CLAUDE.md` records:
  - the Accelerate and encoder measurements
    ([#1](https://github.com/HendrikReh/bgsage/pull/1));
  - the MLX trainer ([#2](https://github.com/HendrikReh/bgsage/pull/2));
  - the analyzer overhead timings
    ([#4](https://github.com/HendrikReh/bgsage/pull/4));
  - the Benchmark PR sample table and the pure-race gap analysis
    ([#2](https://github.com/HendrikReh/bgsage/pull/2),
    [#5](https://github.com/HendrikReh/bgsage/pull/5));
  - the Modal operating notes
    ([#7](https://github.com/HendrikReh/bgsage/pull/7));
  - the post-Crawford auto-double rule, also in `MULTI-PLY.md`
    ([#9](https://github.com/HendrikReh/bgsage/pull/9));
  - the XG referee results and the failed pure-race retraining
    ([#8](https://github.com/HendrikReh/bgsage/pull/8)).
- The build section installs from `build/cmake_accel` and removes the old
  module before copying it, because `cp` over a loaded `.so` makes macOS kill
  the next import ([#12](https://github.com/HendrikReh/bgsage/pull/12)).

[Unreleased]: https://github.com/HendrikReh/bgsage/compare/v2.0.20261118...HEAD
[2.0.20261118]: https://github.com/HendrikReh/bgsage/releases/tag/v2.0.20261118
