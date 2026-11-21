# Engine behavior

NeonGammon has two entry points for different tasks. Use `simulate_game()` or
`simulate_match()` to play out games with a policy. Use `BgBotAnalyzer` to rank
moves, examine equities, or analyze cube decisions. Automatic PureTD selection
belongs to game simulation. The analyzer defaults to cubeful Stage 11 at
1-ply, with optional weights and evaluation levels. [Simulation API](SIMULATION.md)
· [Analyzer configuration](python/neongammon/analyzer.py) · [Model registry](python/neongammon/weights.py)

## Which engine plays the position?

The table describes simulation with the default players. Stage 11 is
the default analyzer model in both the hybrid and the standalone Stage 11
paths.

| Situation | Checker play after the roll | Cube decision before the roll |
| --- | --- | --- |
| Jacoby money game, PureTD available, cube below its cap | PureTD searches one opponent reply with a 0.075 equity filter | Stage 11 at 2-ply |
| Same game once the cube reaches its cap | Stage 11 at 2-ply for the remaining moves | No further doubles |
| Jacoby money game, PureTD unavailable | Stage 11 at 1-ply | Stage 11 at 1-ply |
| Money game without Jacoby | Stage 11 at 1-ply | Stage 11 at 1-ply |
| Match outside the Crawford game | Stage 11 at 1-ply with the match score | Stage 11 at 1-ply with the match score and match equity table |
| Crawford game | Stage 11 at 1-ply with the match score and Crawford state | No doubles |

The simulation calls the Cube policy only when a double is legal. There is no
Cube decision on the opening contest, while the opponent owns the cube, in a
Crawford game, or at the cube cap. After a take, the checker policy receives
the doubled cube value and ownership by the taker. The first roll from a
custom pre-roll position follows ordinary turn rules. [Simulation rules](SIMULATION.md)

Explicit `players=(policy1, policy2)` take precedence. An explicit
`PureTDPolicy` defaults to no lookahead and delegates all Cube decisions to
its analyzer, which defaults to Stage 11 at 2-ply. It also delegates checker
play for match or Crawford contexts, money games without Jacoby, Beaver rules,
and positions at the cube cap. Simulation itself plays without Beavers.
For a forced PureTD move or pass, the single legal successor is returned
directly. [PureTD policy](SIMULATION.md#puretd-checker-policy)

### When PureTD is available

Set `NEONGAMMON_PURETD_LIBRARY` and `NEONGAMMON_PURETD_MODEL` to the native
library and the exported cubeful money checkpoint. In a source checkout, when
both variables are unset, the simulator looks for `build/puretd/libpuretd_cblas.so`
and `build/puretd/cubeful.bin`. The tested filter requires a compatible ABI 7
library that exports `bgc_set_search`. Missing or unloadable files, incompatible
libraries and rejected models use the Stage 11 simulation fallback. An error
from PureTD while selecting a move surfaces to the caller. [Loading and fallback](python/neongammon/puretd.py)

## How Stage 11 evaluates board positions

Stage 11 is a 24-network evaluator. It predicts win, gammon and backgammon
probabilities. Cube-aware evaluation also accounts for ownership and
redoubles. At 1-ply it uses Janowski's cube-life model when the cube is live.
Cube analysis compares No Double, Double/Take and Double/Pass. Match play
uses the players' scores and the match equity table, with Jacoby disabled.
[Cubeful equity](MULTI-PLY.md#3-1-ply-cubeful-evaluation-janowski-interpolation)
· [Match play](MULTI-PLY.md#12-match-play)

| Board situation | Stage 11 evaluation |
| --- | --- |
| Both sides are in their home boards or off, with empty bars | The one-sided bearoff database supplies endgame probabilities when enabled and available. Eligible evaluations and rollout branches can finish from the database. |
| Pure race outside the bearoff database | Dedicated race network inherited from Stage 5. |
| Ordinary contact | The combination of both players' plans selects a network. The contact plans are racing, attacking, priming and anchoring. |
| Backgames and containment | Specialized networks cover deep, middle and double-anchor backgames, plus early containment, containment, snake and massive positions. |

For contact positions, the routing tests snake, containment and massive
before the three backgame categories, then checks early containment before
using the ordinary plan pair. A snake at the root can pin its network for
contact nodes in the search. Race and bearoff positions retain their own
evaluators. This routing applies when Stage 11 evaluates the position,
including Cube decisions in the PureTD hybrid. [Network routing](cpp/src/neural_net.cpp)
· [Bearoff database](CLAUDE.md#bearoff-database) · [Root routing](MULTI-PLY.md#root-routing-pinned-nn-selection)

## Lookahead and move filters

| Mode | What the engine searches |
| --- | --- |
| Stage 11 `1ply` | Static evaluation with the current dice for checker play. Cube analysis evaluates the pre-roll position. |
| Stage 11 `2ply` | One opponent reply after a candidate move, weighted over the opponent's dice. The cubeful checker path starts with up to five candidates within 0.08 equity after 1-ply scoring and can promote moves for deeper evaluation. |
| Stage 11 `3ply` and `4ply` | Two or three further turns. The current cubeful checker path sends all legal moves through the cube-aware tree at these levels. Cubeless search has its own iterative filters. |
| PureTD `checker_plies=0` | Static cubeful scores for all legal candidates. |
| PureTD `checker_plies=1` | One opponent reply. `root_filter=0.0` searches full width. With `root_filter=0.075`, candidates within 0.075 equity of the best static score survive, together with game-ending moves. With a single survivor, the native filter can select it immediately. |

NeonGammon uses the XG naming convention for ply. PureTD's one-reply depth
corresponds to NeonGammon's `2ply`. The two filters use their own scores and
candidate rules. In `BgBotAnalyzer.checker_play()`, use `force_boards` to give a
specific legal post-move board, such as the move actually played, the full
requested evaluation depth even if a filter would otherwise discard it.
[Analyzer and filter dispatch](python/neongammon/analyzer.py) · [Multi-ply search](MULTI-PLY.md)

## Rollouts

Select a rollout level on `BgBotAnalyzer` to compare checker alternatives or
Cube actions through repeated simulations. A truncated rollout stops at a
configured half-move depth and estimates the remaining equity. A full
rollout runs to the end, subject to endgame shortcuts where applicable.
Cubeful trials can end when a double is passed. Rollouts use stratified dice
and variance reduction, and can report standard errors for the estimates.
[Rollout methods](ROLLOUT.md)

| Analyzer level | Default trials | Truncation | Checker play during trials |
| --- | ---: | --- | --- |
| `truncated1` (1T) | 72 | 5 half-moves | 1-ply |
| `truncated2` (2T) | 360 | 7 half-moves | 2-ply for the first decision, then 1-ply. Cube and truncation evaluation use 2-ply. |
| `truncated3` (3T) | 360 | 7 half-moves | 3-ply early, then 2-ply |
| `rollout` (R) | 1,296 | Full by default | 1-ply by default. Depth, trial count and checker/Cube trial policies are configurable. |

Default game simulation uses the policies in the first table. Rollout analysis
is an explicit choice. With default weights, the trial policies use Stage 11
with the configured depths. Checker rollouts filter the move list before
spending trials. 1T filters at 1-ply. 2T, 3T and full rollouts by default
apply a wider 1-ply prefilter followed by a tighter 2-ply filter. When more
than one legal candidate reaches the final filter, the analyzer aims to roll
at least two. Cubeful analysis can also promote promising moves that were
initially filtered out. Cube rollouts evaluate the decision branches from the
pre-roll position. [Rollout candidate selection](ROLLOUT.md#16-best-move-selection-via-rollout)
· [Preset and trial configuration](ROLLOUT.md#17-configuration-reference)
