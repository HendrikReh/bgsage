# Changelog

Changes in this fork ([HendrikReh/bgsage](https://github.com/HendrikReh/bgsage))
on top of upstream [markbgsage/bgsage](https://github.com/markbgsage/bgsage).
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

The fork branched from upstream at `139427d` (2026-10-17, "Add compact
all-roll legal move arrays for video reconstruction"). Everything before that
commit is upstream's history; see `git log 139427d`. The fork targets macOS
on Apple Silicon only (see "Platform Scope" in `CLAUDE.md`). It has no
releases yet, and the package version is still upstream's `2.0.20260907`.

Timings below were measured on an M4 Pro Mac mini (10P+4E cores).

## [Unreleased]

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
    ([#9](https://github.com/HendrikReh/bgsage/pull/9)).

## Not yet on `main`

- **Stage 5's pure-race net in `stage11`, and XG as a Benchmark PR referee**
  (open PR [#8](https://github.com/HendrikReh/bgsage/pull/8)).
  `scripts/export_benchmark_pr_xg.py` exports the sample as one-decision
  `.xg` games for XG's Batch Analyze and scores Sage and GNUbg against both
  the Stage 5 rollout and XG. With XG as referee, the 200h Stage 5 race net
  beats the 100h net `stage11` used, so `stage11` switches to it: Benchmark
  PR 4.11 / 3.17 → 4.10 / 3.14. The PR also adds
  `scripts/run_purerace_sl.py`. Once it is merged, `stage11`'s pure-race
  results in `accuracy_vs_time.py` will move slightly.
