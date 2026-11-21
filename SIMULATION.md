# Recorded game and match simulation

The public package exports `simulate_game`, `simulate_match`, `GameConfig`,
`SimulationCancelled` and `SimulationLimitExceeded`. The package contains the
implementation; no repository scripts, Sage Pro installation, account or network
are needed. Python orchestrates rules and records; the native engine generates
legal moves, detects game results, and evaluates decisions.

```python
import json
from neongammon import BgBotAnalyzer, GameConfig, simulate_game, simulate_match

players = (
    BgBotAnalyzer(eval_level="1ply", parallel_threads=1),
    BgBotAnalyzer(eval_level="2ply", parallel_threads=1),
)
game = simulate_game(seed=42, players=players)
match = simulate_match(match_length=5, seed=42, players=players)
with open("match.json", "w", encoding="utf-8") as f:
    json.dump(match, f)
```

For a custom category, pass a validated pre-roll board through
`GameConfig(start_board=tuple(board), player_on_roll=1, cube_value=2,
cube_owner="player", jacoby=False)`. Boards are mover-relative, with positive
points for the mover, negative for the opponent, nonnegative bars at 25 and 0.
The initial mover is player 1 or 2; cube ownership is relative to that mover.
Explicit boards use ordinary dice, including doubles, on their first turn.
`start_board=None` invokes the standard opening contest: reroll ties, winner
plays both dice, and no opening cube offer. It ignores `player_on_roll`.

Unlimited games default to Jacoby and a maximum cube of 1024. Jacoby suppresses
gammons/backgammons only while the cube remains centered. Pass awards the cube
before doubling; taking doubles the cube and gives ownership to the taker.
Only the owner or either player with a centered cube can offer a double.
Matches start at 0-0, use score-aware evaluations, disable Jacoby, apply Crawford
once (including the one-point match), and resume doubling after Crawford.
Scores cap at the match target, while each game retains raw `points` separately
from `points_awarded`. Low-level `GameConfig` can supply a game's match context;
the complete-match driver currently starts at 0-0.

Beavers/raccoons, resignations and automatic opening doubles are not supported.
The default complete-match budget is 10,000 recorded turns across all games;
`max_total_turns` can change it within a bounded range. This limits transcript
memory as well as computation. Games also have individual `max_turns` limits.
Policy calls explicitly receive `beaver=False`; never interpret records as
simulations with those rules. A turn limit raises `SimulationLimitExceeded`,
without inventing a winner. Caller cancellation raises `SimulationCancelled`;
provide a `cancel` callback to check between decisions. A native evaluation
already in progress must finish before cancellation takes effect.

Results are versioned JSON-compatible dictionaries. Game `turns` retain every
decision's pre-roll board, player, cube and away/Crawford context, cube action,
dice, post-move board and checker-time cube state. Cube context before a double
and checker context after a take are intentionally separate. A pass has no dice
or checker play. Boards remain in that turn's mover perspective; player IDs
remain absolute throughout. Match games contain start and end scores and their
individual replay seeds. Analyze recorded positions separately when reference
analytics, alternative moves, rollout standard errors or PR are needed.

The random generator is isolated per game; a match derives a deterministic
seed for each game. Save seeds, policy settings, model/engine build and rules.
Reproducibility holds within the same engine build and deterministic policies,
not across model upgrades, platforms or floating-point compiler changes. An
external policy is responsible for its own randomness. A seed alone is not
independent validation evidence; split research by complete games or matches.

For bespoke strategies, pass two objects implementing `checker_play` and
`cube_action` with the `BgBotAnalyzer` signatures and result shapes. A policy
can also implement `select_checker_move` with the same position, dice and cube
arguments and return the selected board directly. The simulator checks that
board against the native legal moves. Cube responses are
queried from the responder's policy with the **doubler's perspective**, because
the public cube API's `should_take` describes the opponent's action. Do not flip
that board before returning `should_take`. The API never runs arbitrary code
received over a network. For batches, use processes and independent seeds;
avoid sharing mutable analyzers concurrently. In Sage Pro use Research's paced
worker jobs rather than running large batches inside the API process.

## PureTD checker policy

`simulate_game()` uses PureTD with one opponent reply and a 0.075 equity
filter for Jacoby money games when no `players` are supplied and the native
library and cubeful checkpoint can be loaded. Stage 11 at 2 ply handles
the cube in this hybrid. To make the files available, set
`NEONGAMMON_PURETD_LIBRARY` and `NEONGAMMON_PURETD_MODEL` to their paths before
running the simulation. In a source checkout the simulator also checks
`build/puretd/libpuretd_cblas.so` and `build/puretd/cubeful.bin` when both
variables are unset. Stage 11 handles play when PureTD cannot be loaded or the
game uses match play or non-Jacoby rules. Providing `players` selects the
policies explicitly.

For explicit `PureTDPolicy` instances, the constructor defaults to choosing
the move without lookahead. Pass `checker_plies=1`
to search one opponent reply, equivalent to NeoGammon's 2-ply depth. With one
reply, `root_filter=0.075` searches moves within 0.075 equity of PureTD's best
static estimate and includes game-ending moves. The default `root_filter=0.0`
uses full-width search. A positive filter requires a native library that
exports `bgc_set_search`.

A cubeful Stage 11 analyzer at 2 ply handles doubling, take/pass, and full
analysis by default. The same analyzer chooses the move in match play, under
non-Jacoby or beaver rules, and when the cube is already at its limit.

Supply a local [PureTD C library with ABI 7](https://github.com/alexstrehl/puretd/tree/9c9e5138364a24ab7b907c59262c7255e804e948/c_api)
and a cubeful model exported with PureTD's `export_weights.py`. The tested
checkpoint at that commit is `best_models/cubeful_money_512_512_256_256.pt`.
The policy loads both files directly and can run without importing PyTorch.
For the local experiment, the build and export are in
`logs/puretd_comparison/native`.

```python
from neongammon import BgBotAnalyzer, GameConfig, PureTDPolicy, simulate_game

with PureTDPolicy(
    "logs/puretd_comparison/native/libpuretd_cblas.so",
    "logs/puretd_comparison/native/cubeful.bin",
    checker_plies=1,
    root_filter=0.075,
) as hybrid:
    opponent = BgBotAnalyzer(eval_level="2ply", parallel_threads=1)
    game = simulate_game(seed=42, config=GameConfig(max_cube_value=128),
                         players=(hybrid, opponent))
    opening = game["turns"][0]
    analysis = hybrid.analysis.checker_play(opening["board"], *opening["dice"],
                                            beaver=False, max_cube_value=128)
```

Use `select_checker_move(board, die1, die2, ...)` for a single decision. Use
`analysis.checker_play` to score and rank all legal moves with Stage 11.
