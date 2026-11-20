# Changelog

Changes in NeonGammon ([HendrikReh/neongammon](https://github.com/HendrikReh/neongammon),
called `bgsage` until 2026-11-20) on top of upstream [markbgsage/bgsage](https://github.com/markbgsage/bgsage).
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

The fork branched from upstream at `139427d` (2026-10-17, "Add compact
all-roll legal move arrays for video reconstruction"). Everything before that
commit is upstream's history; see `git log 139427d`. The fork targets macOS
on Apple Silicon only (see "Platform Scope" in `CLAUDE.md`). Releases keep upstream's `2.0.YYYYMMDD` version scheme, dated by the fork's
release. They are published as GitHub releases with Apple Silicon wheels,
not on PyPI.

Timings below were measured on an M4 Pro Mac mini (10P+4E cores).

## [Unreleased]

## [2.0.20261120] - 2026-11-20

### Added

- **`BgBotAnalyzer.cube_decision()`**: the cube decision without the
  cubeless probabilities.
  - At 2-ply and deeper, `cube_action` runs a second, cubeless N-ply search
    for `probs` and `cubeless_equity` after the cube tree has decided.
    `cube_decision` skips it (`decision_only=True` on
    `bgbot_cpp.cube_decision_nply_unified`) and returns a
    `CubeDecisionResult`.
  - Equities and flags are exactly `cube_action`'s. At 1-ply and the
    rollout levels it is `cube_action` minus those fields.
  - `simulate_game` uses it when a policy offers it and keeps `cube_action`
    for custom players. Seeded games are unchanged.
  - Median per call, Stage 11, 20 money positions:

    | Call | `cube_action` | `cube_decision` |
    |---|---|---|
    | 2-ply, 1 thread | 2.21 ms | 0.84 ms |
    | 3-ply, 1 thread | 16.1 ms | 6.3 ms |
    | 3-ply, 4 threads | 9.7 ms | 3.8 ms |
    | 4-ply, 4 threads | 214 ms | 75 ms |

    Five simulated 2-ply money games take 0.74-0.76 s instead of 0.88-0.90 s.
  - `tests/test_cube_decision.py` checks the parity at 1-4 ply for money
    (with and without Jacoby and the beaver), match, Crawford, a cube cap,
    a bearoff position and a root-routed snake, and checks seeded
    simulation records.

### Changed

- **The project is now NeonGammon** (it was `bgsage`). Breaking for imports.
  - Repository: `HendrikReh/bgsage` -> `HendrikReh/neongammon`. GitHub
    redirects the old URLs, clones and remotes; `git remote set-url origin
    https://github.com/HendrikReh/neongammon.git` updates a checkout.
  - Python package and wheel: `bgsage` -> `neongammon`
    (`python/neongammon/`, `from neongammon import BgBotAnalyzer`). There is
    no `bgsage` alias: `import bgsage` fails.
  - Environment variables: `NEONGAMMON_GNUBG`, `NEONGAMMON_SETTINGS` and
    `NEONGAMMON_NO_ROOT_ROUTING`. The `BGSAGE_*` names are still read when
    the new one is unset.
  - Settings file: `~/.config/neongammon/settings.json`. While it does not
    exist, a saved `~/.config/bgsage/settings.json` is read; the next save
    writes the new file (`tests/test_settings.py`).
  - Mobile C API: `bgsage_mobile_*` -> `neongammon_mobile_*`,
    `BgsageMobileEngine` -> `NeongammonMobileEngine`, library
    `libneongammon_mobile.dylib`, CMake option
    `NEONGAMMON_MOBILE_REFERENCE_PYTHON`. The top-level CMake variables are
    `NEONGAMMON_*`.
  - The bot's name in transcripts and reports: player labels "Sage" and
    "Sage Bot" are now "Neongammon"; the benchmark scripts' `SageBot` class
    is `NeongammonBot`.
  - Unchanged: the C++ engine (`bgbot` namespace and headers, the
    `bgbot_cpp` extension, `BgBotAnalyzer`), weights and data, script file
    names such as `run_sage_vs_sage_games.py`, and the measured results
    quoted in the docs. The license stays MPL-2.0, and files from upstream
    keep their copyright notices.

- **Single and batch N-ply checker play run one native routine.**
  - `BgBotAnalyzer.checker_play` (cube-aware N-ply) and `batch_checker_play`
    each had their own implementation of these steps, one in Python and one
    in C++:
    - the 2-ply candidate filter;
    - the cubeless survivor search and its promotions;
    - the cube-aware rescoring;
    - the final top-two promotions.
  - Both now go through `score_checker_position` in `bindings.cpp`. The
    analyzer reaches it through the new `bgbot_cpp.checker_play_nply`, with
    its forced boards, cube cap and thread budget; the cube-tree calls are
    spread over the native pool. The batch runs it per position, and its
    rows may now carry `max_cube_value` and `force_boards`.
  - The Python orchestration of the cube-aware path is gone; the cubeless
    and 1-ply paths are unchanged.
  - Results are identical. On 75 cases, the full move lists equal the old
    Python path's, field for field. The cases cover 2-, 3- and 4-ply, budgets
    of 1, 2 and 4, money, match and Crawford, opponent-owned and capped
    cubes, a root-routed snake, bearoff, a blocked roll and forced boards.
  - Median per call on 24 positions: 2-ply 2.9 -> 2.7 ms (1 thread) and
    2.5 -> 2.2 ms (4 threads); 3-ply unchanged.
  - `tests/test_checker_play_shared_routine.py` holds single and batch to
    the same lists and covers forced boards, the dead cube and the absence
    of the Python pass. It replaces `tests/test_cubeful_full_width.py`.

- **An explicit thread budget of one runs inline, and cheap batches start
  workers only for enough work.**
  - **Budget of one.** An N-ply `BgBotAnalyzer(parallel_threads=1)` was
    raised to two threads, and its cubeful checker play started a
    two-worker pool per call. The money benchmark asks for one thread per
    analysis because it already runs ten analyses at once. Now an explicit
    budget is kept as given: with one, the native search runs serially and
    the per-candidate conversions run inline, with no pool. Pools are also
    skipped when there is a single candidate.
  - **Auto mode for cheap batches.** The 1-ply `batch_evaluate` and
    `batch_post_move_evaluate` cost about 5 us per position, less than
    starting a worker, yet auto mode started one per core. Now it starts one
    worker per `CHEAP_POSITIONS_PER_WORKER` (8) positions. Checker play
    (about 100 us per 1-ply position), N-ply levels and explicit counts are
    unchanged.
  - Median, Stage 11, warm `BatchModels`, 1-ply `batch_evaluate` in auto
    mode:

    | Positions | before | after |
    |---|---|---|
    | 4 | 47 us | 24 us |
    | 16 | 165 us | 97 us |
    | 32 | 188 us | 136 us |
    | 256 | 699 us | 703 us |

  - A 415-decision slice of the money benchmark, ten analyses at once with
    one thread each, gives the same PR in the same time. Its peak thread
    count falls from 47 to 12 at 3-ply and from 20 to 12 at 2-ply.
  - Results never depend on the thread count. This is checked at 1-4
    threads and auto, for the analyzer at 2-ply and 3-ply (money and match)
    and for both batch paths at 1-ply and 2-ply.

- **The analyzer loads each model once on its checker side and shares the
  bearoff database.**
  - A 3-ply `BgBotAnalyzer` loaded Stage 11 three times:
    - for its 1-ply strategy;
    - again inside `create_multipy` for the N-ply strategy;
    - once more for the cube strategy on its first cube action.

    Every analyzer also read the bearoff database from disk.
  - Now the N-ply strategy is built around the 1-ply strategy's base
    (`create_multipy_from_base`), as is the rollout analyzer's 2-ply
    prefilter strategy. The cube strategy keeps its own base, so its
    evaluator configuration stays separate.
  - The database is loaded once per process (`analyzer.load_bearoff_db`) and
    shared with `BatchModels`. `BatchModels` now loads it on first use, so
    an empty batch loads neither a model nor the database.
  - Constructing a 3-ply analyzer takes 17-20 ms instead of 35-41 ms
    (median of 15). First and later analyses are unchanged.
  - `tests/test_analyzer_model_reuse.py` counts the model and database loads
    at each stage. It also compares whole results (checker play, cube
    action, post-move) against the old construction at 2-ply and 3-ply for
    Stage 11, Stage 5, a custom database path and no database, including a
    root-routed snake and a bearoff position.

### Removed

- **Upstream's Windows, Linux and CUDA code.** The fork now builds for Apple
  Silicon only.
  - C++: the CUDA SL trainer (`cuda_nn.cu`, its stub and header) and the
    `cuda_available` / `cuda_supervised_train*` bindings; the MSVC and Win32
    branches in `encoding.cpp`, `multipy.cpp` and `neural_net.cpp`.
  - Build: upstream's CUDA `cpp/CMakeLists.txt` and the MSVC flags.
    `cpp/CMakeLists_cpu.txt` is now `cpp/CMakeLists.txt`.
  - Packaging: the Linux and Windows cibuildwheel targets and classifiers,
    and the upstream-only PyPI job in `wheels.yml`.
  - Python: `neongammon.sl_train` is MLX-only. GNUbg auto-detection looks for
    `gnubg` on PATH and in Homebrew's locations, not the Windows install path.
  - Scripts: the PowerShell scripts, the pywinauto XG driver
    (`xg_batch_win.py`, `xg_folder_chain.py`, `xg_folder_batch.py analyze`),
    the AWS Fargate rollout scripts, and the Windows CUDA-DLL and
    `build_msvc` path setup in about 120 scripts and tests.
  - Docs: the Windows/Linux build instructions and CUDA trainer notes.
  - Kept: the x86-64 AVX2 kernels and CMake branch the Modal rollout image
    (`scripts/modal_rollout.py`) builds on Linux, marked "Modal only".
    Historical numbers keep their "measured upstream on RTX 4070S / Windows"
    labels.

### Fixed

- **1-ply `batch_checker_play` returns the analyzer's list exactly.**
  - The 1-ply batch overloads kept `cubeless_equity` in float and computed
    `equity_diff` with a float subtraction. The analyzer keeps both in
    double, so on 38 of 42 positions the two lists differed in the last
    digits.
  - Both are now double. `tests/test_batch_checker_play_parity.py` holds the
    1-ply batch to the analyzer field for field over its 42 positions (money,
    match, Crawford, a cube on either side, snake, bearoff). It used to
    compare only four positions, without `equity_diff`.

- **Evaluations with and without the bearoff database no longer share cache
  entries.**
  - `BearoffStrategy` reported its base strategy's identity, which keys the
    cubeful engine's per-thread cache and the rollout's shared cache derived
    from it. So on one base strategy, a call with the database and one
    without were served from the same entries, and whichever ran first
    decided both.
  - Example: a bearoff race at 2-ply, centered money cube. On a fresh
    strategy the equity is +0.3049 without the database and +0.1085 with it.
    On a warm thread the second call returned the first call's value.
  - The legacy `MultiPlyStrategy` cache, salted by the base pointer alone,
    had the same flaw for two N-ply strategies around one base.
  - Now `BearoffDB` has a process-unique data identity, new on every load
    or generate. The wrapper mixes it into the base's identity, a wrapper
    over a wrapper of the same database shares that identity, and the
    legacy cache key folds it in too.
  - Normal analyzer and batch calls use one configuration per strategy and
    are unchanged. `tests/test_bearoff_cache_identity.py` covers both call
    orders against fresh-strategy references, distinct database instances,
    nested wrappers and the legacy cache. Four of its tests fail on the
    previous build.

### Documentation

- `CLAUDE.md` records a paired comparison of the N-ply cube tree's interior
  move pick (1-ply cubeful against the primary cube state) against a
  cubeless pick, made with an experiment switch that was not merged. In the
  5-point match benchmark the cubeless pick is clearly worse (total PR
  +0.028 +- 0.009 at 2-ply, +0.029 +- 0.012 at 3-ply; 3-ply checker
  blunders 5 -> 9). In money play the two are level within noise (money
  benchmark 3-ply -0.011 +- 0.007, cube decisions against XG 3-ply
  -0.027 +- 0.014). The engine keeps the cubeful pick.

## [2.0.20261119] - 2026-11-19

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

- **XG as referee for the folder benchmarks**
  (`scripts/export_folder_benchmark_xg.py`, `scripts/score_folder_vs_xg.py`).
  - The export script writes a folder reference's decisions as one-decision
    `.xg` games.
  - `score_folder_vs_xg.py picks` computes Sage's picks. `score` grades
    them against XG and the rollout reference, with paired level
    differences.
  - Grading uses only XG's full-depth values, because XG's shallower values
    are off scale in these positions. `complete` writes the games that let
    XG analyse the remaining picks.
  - First use, the snake folder, XG World Class: 3-ply is the best level
    and 2-ply the worst. Against XG, 3P - 2P is -6.3 +- 1.7 on both
    references and 3P - 1P is -3.3 +- 1.9 / -2.7 +- 1.8. This settles the
    disagreement between the two rollout references in favour of the
    S11-play twin.

### Changed

- **The N-ply batch cube path uses the 1-ply money cube life.**
  - `batch_evaluate`, `batch_cube_action` and `batch_post_move_evaluate`
    build no cube tree. At every depth they apply Janowski once, at the root,
    to cubeless probs. They used the analyzer's 1-ply money cube life at
    1-ply and the engine's 0.68 from 2-ply on, although `CLAUDE.md` said
    "at the matching levels".
  - The analyzer's N-ply values were fitted for the cube tree's 1-ply
    leaves. The 1-ply value was fitted for exactly this construct, and
    measured against XG it fits best.
  - Measured on the money benchmark's 2,196 cube positions (2,842
    decisions), `batch_cube_action` cube PR against XG, blunders in
    brackets:

    | Depth | 0.68 | 1-ply value | 2-ply value |
    |---|---|---|---|
    | 2-ply | 2.33 (19) | 1.00 (14) | 1.20 (22) |
    | 3-ply | 1.97 (17) | 1.04 (19) | 1.15 (20) |
    | 4-ply | 1.98 (12) | 0.94 (12) | 0.95 (15) |

    The paired difference to 0.68 is -1.33 +- 0.12, -0.93 +- 0.14 and
    -1.04 +- 0.11. The gain holds on both seed halves and against the Sage
    reference. Against the 2-ply value, the 1-ply value is ahead at 2-ply
    (-0.19 +- 0.08) and 3-ply (-0.11 +- 0.05) and tied at 4-ply.
  - Now these paths use the 1-ply value at every depth
    (`ROOT_JANOWSKI_CUBE_LIFE_PLY`), for money positions only.
    `scripts/score_batch_cube_life_xg.py` reproduces the table.
  - This changes the N-ply batch cube decisions and cubeful equities of
    money contact positions. Match play, races and `batch_checker_play` are
    unaffected.

- **`cube_action(incl_2ply_details=True)` runs on the evaluation engine.**
  - The per-roll details came from an older recursion in `cube.cpp` that
    differed from the engine behind the plain call in three ways:
    - it picked moves by 1-ply cubeless equity at the captured levels;
    - it called the network one candidate at a time, with the lookup
      sigmoid;
    - it had no root routing.
  - Its headline `equity_nd` / `equity_dt` therefore drifted from the plain
    call's, by up to 0.07 at 2-ply and 0.09 at 3-ply on 76 random
    positions. `CLAUDE.md` claimed they matched.
  - The details are now recorded from the very tree the plain call walks,
    with a per-roll capture at its top two levels. So:
    - the headline and the decision equal the plain call's exactly, at
      every depth and for money and match alike;
    - each ND player-roll equity equals the opponent's standalone
      (N-1)-ply cube analysis of that board (from 3-ply on, exactly);
    - the boards are the engine's picks: 1-ply cubeful against the ND cube
      state, with root routing.
  - This changes results. On the 76 positions (half 5-away/3-away), 4-7% of
    the per-roll boards changed. ND per-roll equities, and so `roll_luck`
    values, moved by a median of 0.001; the 95th percentile was 0.003 at
    2-ply and 0.026 at 3-ply.
  - The details call now costs what a plain call does. Median, Stage 11,
    4 threads, machine idle:

    | Level | plain | details before | details after |
    |---|---|---|---|
    | 2-ply | 1.7 ms | 1.9 ms | 1.6 ms |
    | 3-ply | 9.3-9.7 ms | 11.9-12.0 ms | 9.6-9.7 ms |
    | 4-ply | 130-131 ms | 165-166 ms | 130 ms |

  - `tests/test_2ply_details.py` now checks the details against the plain
    call (exactly) and against standalone analyses of each roll's board,
    for money, match, Crawford, an opponent-owned cube and a root-routed
    snake.

- **One implementation of the dice rules in the move generator.**
  - The two sorted `possible_boards` overloads repeated the same traversal
    word for word, and `possible_boards_unsorted` repeated the rules once
    more:
    - doubles play as many dice as they can;
    - otherwise both dice, else the larger, else the smaller;
    - with no legal move, the player stands pat.
  - Now one `apply_dice_rules` in `moves.cpp` drives both generators, and
    the returning overload delegates to the buffer one. Maintenance only:
    the output is byte-identical and the timings are unchanged within noise.
  - A before-and-after corpus checks this: 413 boards, every ordered roll,
    14,760 cases with 304,354 boards. It covers stand-pats, the larger- and
    smaller-die rules, partial doubles, bear-offs, wins mid-roll and hits.
  - New binding `bgbot_cpp.possible_moves_unsorted` exposes the hot paths'
    first-seen order. `tests/test_move_generator.py` pins both outputs with
    a digest computed before the change.

- **The legacy N-ply position cache is allocated on first use.**
  - `MultiPlyStrategy`'s thread-local `PosCache` has 262,144 entries of
    136 bytes, about 36 MB per thread. Only the old cubeless recursion
    fills it: the hybrid evaluator and full-depth-opponent mode. The
    default tree goes through the cubeful engine and never does.
  - Even so, every thread that called `clear_cache()` allocated the table
    and zeroed it on each clear. That includes every rollout worker, which
    clears it at the start of its trials.
  - Now the table is allocated on its first insert, and a clear zeroes it
    only when something was inserted since the last one.
  - Results are byte-identical between the two builds on:
    - the hybrid and full-depth evaluators, which do use the cache;
    - 3-ply checker play;
    - truncated-rollout cube analysis.

    `tests/test_legacy_pos_cache.py` checks that the cache still fills,
    hits and clears.
  - Peak memory, Stage 11, two positions (cube action plus checker play):

    | Level | before | after |
    |---|---|---|
    | `truncated1`, 10 threads | 1,013-1,026 MB | 503-509 MB |
    | `truncated1`, 2 threads | 706-710 MB | 436-438 MB |
    | `3ply`, 10 threads | 349-351 MB | 312-320 MB |

  - A warm `clear_cache()` takes 0.03-0.10 ms instead of 0.20 ms.
  - Elapsed times are unchanged within noise. They were measured under
    load from another job (load average ~32).

- **Batch calls load each model once, and can keep it loaded**
  (`BatchModels`).
  - Every batch function used to build its strategies from the weight
    files on every call: ~15 ms for Stage 11, paid even by an empty batch.
    N-ply checker play paid it twice, because its N-ply strategy loaded a
    second copy of the base.
  - Now:
    - an empty batch checks the level and returns without loading
      anything;
    - N-ply strategies are built around the batch's base, through the new
      `bgbot_cpp.create_multipy_from_base`;
    - all model types go through `create_strategy`.
  - A caller-owned `BatchModels` keeps the strategies for later calls. Pass
    it as `models=` to `batch_evaluate`, `batch_post_move_evaluate`,
    `batch_cube_action` or `batch_checker_play`.
  - Results are unchanged. `tests/test_batch_models.py` holds them to
    strategies built the old way for Stage 11 and Stage 5 at 1-ply and
    2-ply, counts the factory calls, and runs concurrent calls on one
    shared `BatchModels`.
  - Median per call, Stage 11, 10 threads, measured under load from
    another job (load average ~28):

    | Call | before | after | reused `BatchModels` |
    |---|---|---|---|
    | empty batch, 1-ply / 2-ply checker | 20-27 / 65-77 ms | 0.01 / 0.03 ms | 0.00 ms |
    | 1-ply pre-roll, 1 position | 29-32 ms | 20-31 ms | 0.01 ms |
    | 2-ply checker, 1 position | 72-83 ms | 40-45 ms | 3.4 ms |

    Batches of 32 and more are dominated by the search, and under this load
    their timings are noise.

- **Cubeful checker play at 3-ply and deeper skips the cubeless pre-pass.**
  - At these depths every legal move goes through the cube-aware tree,
    which replaces each candidate's probs, both equities and depth label.
    The cubeless filter-and-search pass that ran first decided nothing, so
    the wrapper now only enumerates the legal boards.
  - Results are unchanged: the full ordered move list matches the old path
    field for field (`tests/test_cubeful_full_width.py`: contact, bearoff,
    a blocked roll, a root-pinned snake, Crawford, capped and
    opponent-owned cubes, `force_boards`).
  - Measured on 24 random positions with the CPU loaded by another job, so
    only the ratios hold: 3-ply total time -17% with 2 threads and -14% with
    10 threads (median -18% / -37%), 4-ply -14% (8 positions, 10 threads).
  - 2-ply keeps the pass: its filter and promotion pick which moves the
    cube-aware tree searches.

- **Cubeful 2-ply checker play now searches only the 1-ply filter's
  survivors** (`CUBEFUL_FILTERED_PLIES` in `analyzer.py`).
  - It used to score every legal move in the cube-aware 2-ply tree. In
    positions with many moves the pick then often fell on a move the filter
    had pruned and the 2-ply search over-valued.
  - Pruned moves keep 1-ply Janowski values and enter the search only
    through the stale-top-two promotion, as in cubeless N-ply and
    `batch_checker_play`.
  - Snake folder, paired against the old 2-ply: checker PR -2.2 to -2.9
    (+- 0.7) under XG and both rollout references.
  - Neutral on the money benchmark (-0.016 +- 0.022), the 5-point match
    benchmark (+0.001 +- 0.012) and the backgame folders. 2-ply checker
    play runs 2.2-2.4x faster.
  - 3-ply and deeper still score every move: with the filter, the money
    benchmark's 3-ply checker PR rose 0.59 -> 0.64.

- **The 2-ply money contact cube life now rises with the average pip count
  too**: x = 0.7625 + 0.00125 x (average pips - 120), about 0.71 at 75 pips
  and 0.80 at 150, instead of a flat 0.80. Every leaf of the 2-ply tree takes
  its own pip count.
  - Fitted to XG World Class's cube decisions on the money benchmark's 2,196
    cube positions. `scripts/fit_contact_cube_life.py` now plays every grid
    point through the engine, so it scores the leaves' own values at 2-ply.
  - Fitted on half the seeds and scored on the other half, against a flat
    cube life fitted the same way: 2-ply cube PR -0.19 +- 0.08 against XG
    and -0.22 +- 0.09 against the Sage reference; cube blunders against XG
    16 -> 10.
  - Money benchmark at 2-ply: cube PR 1.31 -> 1.04 (cube blunders 19 -> 12),
    checker play unchanged.
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

- **The batch functions use the bearoff database, as the analyzer does.**
  - `BgBotAnalyzer` loads the one-sided bearoff database by default; the
    batch functions never attached it, so the two disagreed on bearoff
    positions.
  - On 50 bearoff and near-bearoff positions, `batch_checker_play` picked a
    different best move than the analyzer on 6-8 of them (1-, 2- and
    3-ply), with equities up to 0.59 apart. The 1-ply `batch_evaluate` and
    `batch_post_move_evaluate` were up to 0.20 and 0.11 off.
  - `BatchModels` now attaches the database by default, as `BgBotAnalyzer`
    does: `bearoff_db=True`, a path, or `False`. The batch functions take
    `bearoff_db=` with `weights`.
    - The 1-ply strategy is wrapped in `BearoffStrategy`.
    - The N-ply strategies have the database attached to their search.
    - The database is loaded once per process (1.1 ms); a batch call with
      it costs about 0.35 ms more.
  - Batch and analyzer now agree exactly on those positions.
    `tests/test_batch_checker_play_parity.py` holds the full move lists to
    the analyzer's with the database and without, now with bearoff fixtures.
  - Contact positions are unaffected. Bearoff positions get the database's
    exact values unless `bearoff_db=False` is passed.

- **N-ply `batch_checker_play` returns the analyzer's list.**
  - Both N-ply overloads ran the cube-aware tree for every legal move. At
    2-ply the pick could then land on a move the analyzer's 1-ply filter
    rejects: the filter that
    [#26](https://github.com/HendrikReh/neongammon/pull/26) added for the
    analyzer never reached the batch.
    Three more differences, at every depth:
    - the probs came from the cubeless tree;
    - the cube tree was routed per node, not from the decision's root;
    - match positions were searched at cube value 1.
  - On 40 random positions with mixed cube states, the best move differed
    on 4 at 2-ply, and equities differed by up to 0.71 at 2-ply and 0.33 at
    3-ply.
  - The two overloads now share one routine, `batch_checker_play_nply` in
    `bindings.cpp`. It takes the analyzer's steps one by one:
    1. the filter;
    2. the cubeless search with its promotion;
    3. the cube tree for the searched moves, with 1-ply Janowski for the
       rest;
    4. the cubeful promotion, one cube-tree call per promoted move.
  - The filtered depths come from Python (`cubeful_filtered_plies`, set to
    `CUBEFUL_FILTERED_PLIES`).
  - `tests/test_batch_checker_play_parity.py` holds the batch to
    `BgBotAnalyzer.checker_play` (bearoff database off; the batch attaches
    none), field for field. It covers 39 positions at 2-ply and 15 at 3-ply,
    including:
    - money, match and Crawford positions;
    - a cube on either side;
    - a snake under root routing;
    - promotions that end up best.
  - Running the cube tree on fewer moves makes it faster. Measured under
    load from another job (load average 8-15), Stage 11, 10 threads:

    | Batch | before | after |
    |---|---|---|
    | 2-ply, 32 positions | 0.21-0.23 s | 0.067-0.069 s |
    | 2-ply, 256 positions | 1.02-1.07 s | 0.20-0.21 s |
    | 3-ply, 32 positions | 2.08-2.10 s | 1.91-1.93 s |

    At 3-ply the cubeless search, whose values the cube tree replaced, no
    longer runs.

- **Batch functions apply Jacoby and the beaver per position.**
  - `batch_evaluate`, `batch_cube_action`, `batch_post_move_evaluate` and
    `batch_checker_play` used to switch Jacoby (and in `batch_evaluate`
    the beaver) off for the whole batch as soon as one row was a match.
    A money position's result then depended on the rest of the batch.
    Example: the starting position, 1-ply, Stage 11:
    - `batch_evaluate` cubeful equity: +0.1133 alone, +0.1181 beside a
      5-away/3-away row;
    - its Double/Take equity lost the beaver (DB −0.4508 became DT
      −0.2254);
    - `batch_checker_play`'s best move: +0.2359 alone, +0.2472 in the
      mixed batch.
  - The native batch functions already apply both rules to money positions
    only, so the flags are now passed through unchanged. Batches without a
    match row are unaffected.
  - `tests/test_batch_mixed_rules.py` checks every row of mixed batches,
    in both orders, against the same row alone at 1-ply and 2-ply. It needs
    no benchmark files.

- **The test suite runs clean on macOS**
  ([#16](https://github.com/HendrikReh/neongammon/pull/16)): 318 passed,
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
  ([#17](https://github.com/HendrikReh/neongammon/pull/17)). All 291,242 rows
  were rolled out on Modal (~$223), and the retrained net's holdout ER fell
  (general rows 36.8 → 24.4), but its decisions did not improve. On the money
  benchmark's 237 containment-routed decisions it scores 3.50 / 2.05 / 1.28
  at 1P / 2P / 3P against 3.06 / 1.94 / 1.18 for the installed net, and the
  containment folder is a tie. The installed net stays.
- The `scripts/accuracy_vs_time.py` results were re-measured after the
  race-net swap ([#8](https://github.com/HendrikReh/neongammon/pull/8)).
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
  [#2](https://github.com/HendrikReh/neongammon/pull/2)). `supervised_train` and
  `supervised_train_preencoded` take the same arguments and return the same
  dict as the CUDA bindings. They use CUDA when available and otherwise Apple
  MLX on the Mac GPU (`pip install mlx`). All 20 training scripts now call
  them. The MLX path transcribes `cuda_nn.cu`. It is not bit-identical to
  CUDA, but `tests/test_sl_train.py` holds it to a float64 NumPy
  transcription within 2e-5. On the snake-net recipe it reaches holdout ER
  53.96, against 53.95 for CUDA on an RTX 4070S.
- **Configurable GNUbg binary** (`bgsage.settings`,
  [#1](https://github.com/HendrikReh/neongammon/pull/1)). The path is resolved
  from, in order: `set_gnubg_path()` or a per-instance `gnubg_path=`, the
  `BGSAGE_GNUBG` environment variable, `~/.config/bgsage/settings.json`, and
  auto-detection on `PATH`. `GnuBgAnalyzer` and `post_move_analytics` now run
  on macOS.
- **Cloud rollouts on Modal** (`scripts/modal_rollout.py`,
  [#7](https://github.com/HendrikReh/neongammon/pull/7)). The script rolls out
  position files on Modal CPU containers in the SL trainers' format, with a
  `.meta.jsonl` sidecar. Runs are resumable, and `--max-usd`, `--chunk` and
  `--local` are available. Measured cost is about $0.002 per containment
  board and $0.0001 per pure-race board.
- **Pure-race position harvest** (`scripts/harvest_race_positions.py`,
  [#7](https://github.com/HendrikReh/neongammon/pull/7)). 30,000 self-play games
  yield 329,753 train and 36,478 holdout race boards, as the basis for
  retraining the PureRace net on rollout targets.
- **Benchmark PR sampling** (`generate_benchmark_pr.py --resume
  --decision-step N`, [#1](https://github.com/HendrikReh/neongammon/pull/1)).
  This rolls out every Nth saved decision. The fork's reference is the
  1-in-20 sample: 6,944 decisions and 20,302 positions, about 4.5 hours.
  Both PR scorers accept `--decision-step N` and score exactly that sample.
- **CHANGELOG** for the fork ([#10](https://github.com/HendrikReh/neongammon/pull/10)).
- **`score_benchmark_pr_strategy` binding**
  ([#2](https://github.com/HendrikReh/neongammon/pull/2)).
  `score_benchmark_pr.py` now scores every registered model type: 5-NN,
  pair, and backgame-aware models including `stage11`.
- **Accuracy against thinking time** (`scripts/accuracy_vs_time.py`,
  [#10](https://github.com/HendrikReh/neongammon/pull/10)). The
  script measures Benchmark PR and single-thread CPU time per decision for
  Open Sage and GNUbg at 1–4 ply, with both engines picking moves as in
  normal use and depths in the XG convention. First results (`stage11`,
  5,140 decisions), PR at roughly equal CPU time: 2-ply 3.18 vs 3.44, 3-ply
  2.53 vs 2.75, 4-ply 2.55 vs 2.56. Open Sage 3-ply (~30 ms) matches GNUbg
  4-ply (~145 ms). The reference is Stage 5's rollout, which leans towards
  Open Sage.
- **XG as a Benchmark PR referee** (`scripts/export_benchmark_pr_xg.py`,
  [#8](https://github.com/HendrikReh/neongammon/pull/8)). The script exports the
  sample as one-decision `.xg` games for XG's Batch Analyze, reads XG's
  ranked moves back, and scores Sage and GNUbg against both the Stage 5
  rollout and XG. Against XG 4-ply, `stage11` 2-ply (before the race-net
  swap below) scores 1.24 and GNUbg
  2-ply 1.40 (paired difference −0.17 ± 0.07). The PR also adds
  `scripts/run_purerace_sl.py`, the pure-race fine-tuning experiment, which
  did not beat Stage 5's net.
- **Analysis tools** ([#6](https://github.com/HendrikReh/neongammon/pull/6)):
  - `scripts/race_gap_analysis.py` breaks down pure-race PR into bearoff
    positions and races outside the database.
  - `scripts/race_net_swap.py` scores the same races with other race nets
    swapped in.
  - `scripts/bench_analyzer_overhead.py` measures per-call latency of
    `cube_action` and `checker_play`.
- **Tests:**
  - `tests/test_encoding_golden.py` pins the encoder features on 1,000
    positions ([#1](https://github.com/HendrikReh/neongammon/pull/1)).
  - `tests/test_settings.py` covers the GNUbg path precedence
    ([#1](https://github.com/HendrikReh/neongammon/pull/1)).
  - `tests/test_analyzer_thread_limits.py` pins thread limits per level
    ([#3](https://github.com/HendrikReh/neongammon/pull/3)).
  - `tests/test_cube_strategy_threads.py` calls one analyzer from 8 threads
    and requires results bit-identical to a fresh analyzer per call
    ([#6](https://github.com/HendrikReh/neongammon/pull/6)).

### Changed

- **`stage11` uses Stage 5's 200h pure-race net**
  ([#8](https://github.com/HendrikReh/neongammon/pull/8)) instead of the 100h net
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
  ([#1](https://github.com/HendrikReh/neongammon/pull/1)). All batched candidate
  kernels go through `NeuralNetwork::forward_group`. On Apple arm64 it runs
  one `cblas_sgemm` per layer on the CPU's matrix unit (SME/AMX). It uses the
  exact Cephes sigmoid instead of the clamped NEON lookup table, so equities
  move by up to about 0.004. The path is on by default;
  `-DBGBOT_USE_ACCELERATE=OFF` restores the portable kernels, which are
  bit-identical to the previous engine.
- **Faster contact encoder** ([#1](https://github.com/HendrikReh/neongammon/pull/1)).
  `hitting_shots`, `double_hitting_shots`, `compute_piploss`, `moment2` and
  `backbone` use bitmask formulations. That takes 1.80 → 1.17 µs per board,
  bit-identical on 655k boards. Together with Accelerate, `checker_play` is
  1.37–1.41x faster at 2- and 3-ply. Money benchmark PR is unchanged (2-ply
  1.67 → 1.68, 3-ply 0.60).
- **Analyzer call overhead** ([#3](https://github.com/HendrikReh/neongammon/pull/3)).
  `cube_action` loads its base strategy once per analyzer instead of
  reloading all weight files (24 for `stage11`) on every call. The 1-ply
  cubeful `checker_play` conversion runs serially instead of in a thread pool
  (`SERIAL_CUBEFUL_1PLY`). Median speed-up per call: `cube_action` 1-ply
  about 1,500x, 2-ply about 13x, 3-ply about 2.3x, and `checker_play` 1-ply
  about 8x. Code that builds a new analyzer per call still pays the weight
  load.
- **`score_benchmark_pr.py` uses the bearoff database**, as the engine and
  GNUbg do ([#5](https://github.com/HendrikReh/neongammon/pull/5)). Only the
  pure-race column moves: `stage11` 1-ply 4.15 → 4.11, 2-ply 3.22 → 3.17.
  `--no-bearoff` scores the raw networks.
- **N-ply PR scoring adds the PubEval opponent prefilter** that
  `BgBotAnalyzer` uses ([#2](https://github.com/HendrikReh/neongammon/pull/2)).
  Stage 5 2-ply goes from 4.104 to 4.097.
- **Repo-root paths** ([#1](https://github.com/HendrikReh/neongammon/pull/1)).
  `generate_benchmark_pr.py`, `score_benchmark_pr_gnubg.py` and
  `test_match_gnubg_comparison.py` resolve `python/`, `build/`, `data/` and
  `logs/` inside this repo instead of the host project above it.
- **`test_match_gnubg_comparison.py` evaluates through `BgBotAnalyzer`**
  ([#1](https://github.com/HendrikReh/neongammon/pull/1)), so it runs with any
  model, not only 5-NN ones.

### Fixed

- **Post-Crawford cube for the trailer**
  ([#9](https://github.com/HendrikReh/neongammon/pull/9)). After the Crawford
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
  ([#3](https://github.com/HendrikReh/neongammon/pull/3)). `parallel_threads=1`
  was replaced by the CPU count (`workers <= 1` instead of `workers <= 0`),
  and `_OnePlyAnalyzer` never received the limit.
- **GNUbg comparison without a cube decision**
  ([#1](https://github.com/HendrikReh/neongammon/pull/1)). At double match point,
  in the Crawford game and for the post-Crawford leader, GNUbg prints no cube
  analysis. These cases now count as agreement when bgsage also has no cube
  action, which takes the tally from 13 / 0 / 5 skipped to 18 agree.
- **Benchmark PR scoring counts only the rolled-out sample**
  ([#2](https://github.com/HendrikReh/neongammon/pull/2)). The scorers used to
  include decisions whose candidates happened to be covered, a skewed set
  (Stage 5 2-ply 3.37 on the sample against 5.27 on those extras).

### Documentation

- The fork is declared Apple Silicon only in `README.md` and `CLAUDE.md`,
  with a tested Apple Silicon build section. Windows, Linux and CUDA builds
  are kept but not supported
  ([#1](https://github.com/HendrikReh/neongammon/pull/1)).
- `CLAUDE.md` records:
  - the Accelerate and encoder measurements
    ([#1](https://github.com/HendrikReh/neongammon/pull/1));
  - the MLX trainer ([#2](https://github.com/HendrikReh/neongammon/pull/2));
  - the analyzer overhead timings
    ([#4](https://github.com/HendrikReh/neongammon/pull/4));
  - the Benchmark PR sample table and the pure-race gap analysis
    ([#2](https://github.com/HendrikReh/neongammon/pull/2),
    [#5](https://github.com/HendrikReh/neongammon/pull/5));
  - the Modal operating notes
    ([#7](https://github.com/HendrikReh/neongammon/pull/7));
  - the post-Crawford auto-double rule, also in `MULTI-PLY.md`
    ([#9](https://github.com/HendrikReh/neongammon/pull/9));
  - the XG referee results and the failed pure-race retraining
    ([#8](https://github.com/HendrikReh/neongammon/pull/8)).
- The build section installs from `build/cmake_accel` and removes the old
  module before copying it, because `cp` over a loaded `.so` makes macOS kill
  the next import ([#12](https://github.com/HendrikReh/neongammon/pull/12)).

[Unreleased]: https://github.com/HendrikReh/neongammon/compare/v2.0.20261120...HEAD
[2.0.20261120]: https://github.com/HendrikReh/neongammon/compare/v2.0.20261119...v2.0.20261120
[2.0.20261119]: https://github.com/HendrikReh/neongammon/compare/v2.0.20261118...v2.0.20261119
[2.0.20261118]: https://github.com/HendrikReh/neongammon/releases/tag/v2.0.20261118
