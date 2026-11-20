# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Batch position evaluation for the Open Sage bot engine.

Evaluates lists of positions in parallel, returning cubeless probabilities,
cubeless equity, cubeful equity, and (for pre-roll) cube decisions.

Parallelization happens in C++ across positions.  Individual multi-ply
evaluations run serially so there is no double-parallelization.

Typical usage::

    from bgsage.batch import batch_evaluate, batch_post_move_evaluate

    # Pre-roll positions (with cube decisions)
    positions = [
        {"board": STARTING_BOARD, "cube_value": 1, "cube_owner": "centered"},
        {"board": other_board,    "cube_value": 2, "cube_owner": "player"},
    ]
    results = batch_evaluate(positions, eval_level="3ply")
    for r in results:
        print(f"Eq: {r.cubeless_equity:+.3f}  CF: {r.cubeful_equity:+.3f}")

    # Post-move positions (after a player moves, before opponent rolls)
    post_positions = [
        {"board": post_move_board1, "cube_owner": "centered"},
        {"board": post_move_board2, "cube_owner": "player"},
    ]
    results = batch_post_move_evaluate(post_positions, eval_level="1ply")
    for r in results:
        print(f"CL={r.cubeless_equity:+.3f}  CF={r.cubeful_equity:+.3f}")

Each call loads the model's weights (~15 ms for Stage 11, twice for N-ply
checker play before the base was shared). Callers that submit many small
batches load them once instead by passing a :class:`BatchModels`::

    models = BatchModels()            # production model; loads lazily
    for chunk in chunks:
        batch_checker_play(chunk, eval_level="2ply", models=models)
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import bgbot_cpp

from .analyzer import (
    CUBEFUL_FILTERED_PLIES, _contact_cube_life, _contact_cube_slope, load_bearoff_db,
    resolve_owner,
)
from .types import CheckerPlayResult, CubeActionResult, MoveAnalysis, PostMoveAnalysis, Probabilities
from .weights import WeightConfig, WeightConfigPair, bearoff_db_path, default_weights

_LEVELS = ("1ply", "2ply", "3ply", "4ply")

#: Cube life of the pre-roll and post-move batch paths: at every depth they
#: apply Janowski once, at the root, to cubeless probs (N-ply probs from
#: 2-ply on), with no cube tree. That is the construct the analyzer's 1-ply
#: money value was fitted for; the N-ply values were fitted for the cube
#: tree's 1-ply leaves instead. Measured 2026-11-19 against XG on the money
#: benchmark's 2,196 cube positions (scripts/score_batch_cube_life_xg.py),
#: batch_cube_action cube PR with the engine's 0.68 / the 1-ply value / the
#: 2-ply value: 2-ply 2.33 / 1.00 / 1.20, 3-ply 1.97 / 1.04 / 1.15, 4-ply
#: 1.98 / 0.94 / 0.95. Money only, like the analyzer's (the native side
#: applies contact_x to money positions alone).
ROOT_JANOWSKI_CUBE_LIFE_PLY = 1


#: Auto mode (``n_threads=0``) of the 1-ply pre-roll and post-move batches
#: starts one worker per this many positions. Each position costs about 5 us
#: there, less than starting and joining a worker, so the native default (one
#: worker per core) made small batches slower. Measured 2026-11-19 (Stage 11,
#: M4 Pro, warm BatchModels), median us: 4 positions 21 serial / 56 auto,
#: 8: 37 / 78, 32: 142 serial / 96 at 4 workers / 153 auto, 128: 755 / 290 at
#: 8 workers; with 8 per worker every size is within 7% of its best count.
#: Checker play (~100 us per 1-ply position) and N-ply evaluations keep the
#: native default, which already caps workers at the batch size.
CHEAP_POSITIONS_PER_WORKER = 8


def _auto_threads(n_threads: int, n_positions: int, n_plies: int) -> int:
    """The thread count for a pre-roll / post-move batch: an explicit count
    as given; in auto mode at 1-ply, one worker per
    ``CHEAP_POSITIONS_PER_WORKER`` positions, at most one per core."""
    if n_threads > 0 or n_plies > 1:
        return n_threads
    return max(1, min(os.cpu_count() or 1, n_positions // CHEAP_POSITIONS_PER_WORKER))


def _root_cube_life_kwargs() -> dict:
    return dict(contact_x=_contact_cube_life(ROOT_JANOWSKI_CUBE_LIFE_PLY),
                contact_x_pip_slope=_contact_cube_slope(ROOT_JANOWSKI_CUBE_LIFE_PLY))


def _check_level(eval_level: str) -> int:
    """The depth of a batch evaluation level; raises ValueError for any other."""
    if eval_level not in _LEVELS:
        raise ValueError(
            f"Unsupported eval_level for batch: {eval_level!r}. "
            f"Use '1ply', '2ply', '3ply', or '4ply'."
        )
    return int(eval_level[0])


class BatchModels:
    """Loaded strategies for the batch functions, owned by the caller.

    Every batch function builds the strategies it needs from ``weights``
    unless it is given a ``BatchModels``; then it takes them from here, and
    they stay loaded for the next call. The base (1-ply) strategy is loaded
    once, on first use, and every N-ply strategy is built around it, so no
    weights are loaded twice. Results are the same either way.

    ``bearoff_db`` works as in :class:`~bgsage.BgBotAnalyzer`: True (the
    default) loads the one-sided bearoff database from ``data/``, a string
    loads it from that path, False leaves it out. With it, the 1-ply strategy
    returns the database's exact probs at bearoff positions and the N-ply
    strategies stop their search there, as the analyzer's do; a missing
    file leaves it out silently, as the analyzer does.

    Strategies are immutable once built and safe to share between threads
    and between concurrent batch calls; their N-ply caches are per thread.
    """

    def __init__(self, weights: WeightConfig | WeightConfigPair | None = None,
                 bearoff_db: bool | str = True):
        self.weights = default_weights() if weights is None else weights
        self._db_path = bearoff_db if isinstance(bearoff_db, str) else (
            bearoff_db_path() if bearoff_db else None)
        self._lock = threading.Lock()
        self._raw_base = None
        self._base = None
        self._multipy: dict[tuple, object] = {}

    @property
    def bearoff_db(self):
        """The bearoff database, loaded on first use (shared per process), or
        None when off or missing."""
        return load_bearoff_db(self._db_path) if self._db_path else None

    def base(self):
        """The 1-ply strategy, loaded on first use (bearoff-aware when the
        database is loaded)."""
        with self._lock:
            if self._base is None:
                w = self.weights
                self._raw_base = bgbot_cpp.create_strategy(
                    w.strategy_type, w.weight_paths_list, w.hidden_sizes_list)
                self._base = (self._raw_base if self.bearoff_db is None else
                              bgbot_cpp.BearoffStrategy(self._raw_base, self.bearoff_db))
            return self._base

    def multipy(self, n_plies: int, filter_max_moves: int = 5,
                filter_threshold: float = 0.08):
        """The N-ply strategy for these settings, built around :meth:`base`.

        Serial (``parallel_evaluate=False``): the batch functions parallelise
        across positions.
        """
        key = (n_plies, filter_max_moves, float(filter_threshold))
        self.base()
        with self._lock:
            strategy = self._multipy.get(key)
            if strategy is None:
                # Around the raw base, with the database attached to the
                # search itself, as BgBotAnalyzer builds its N-ply strategy.
                strategy = bgbot_cpp.create_multipy_from_base(
                    self._raw_base, n_plies=n_plies,
                    filter_max_moves=filter_max_moves,
                    filter_threshold=filter_threshold,
                    parallel_evaluate=False, parallel_threads=0)
                if self.bearoff_db is not None:
                    bgbot_cpp.multipy_set_bearoff_db(strategy, self.bearoff_db)
                self._multipy[key] = strategy
            return strategy


def _resolve_models(weights, models, bearoff_db) -> BatchModels:
    if models is None:
        return BatchModels(weights, bearoff_db=True if bearoff_db is None else bearoff_db)
    if weights is not None and weights is not models.weights:
        raise ValueError("pass weights or models, not both")
    if bearoff_db is not None:
        raise ValueError("pass bearoff_db to BatchModels, not with models=")
    return models


@dataclass
class PositionEval:
    """Result of evaluating a single pre-roll position."""

    probs: Probabilities
    cubeless_equity: float
    cubeful_equity: float
    equity_nd: float
    equity_dt: float
    equity_dp: float
    should_double: bool
    should_take: bool
    optimal_action: str
    is_beaver: bool = False


def batch_evaluate(
    positions: list[dict],
    eval_level: str = "1ply",
    weights: WeightConfig | WeightConfigPair | None = None,
    *,
    n_threads: int = 0,
    filter_max_moves: int = 5,
    filter_threshold: float = 0.08,
    jacoby: bool = True,
    beaver: bool = True,
    models: BatchModels | None = None,
    bearoff_db: bool | str | None = None,
) -> list[PositionEval]:
    """Evaluate a batch of pre-roll positions in parallel.

    Each position dict must contain:
        board: list[int]  — 26-element board from mover's perspective
        cube_value: int   — current cube value (1, 2, 4, ...)
        cube_owner: str   — "centered", "player", or "opponent"

    Optional match play fields:
        away1: int        — points player needs (0 = money game)
        away2: int        — points opponent needs (0 = money game)
        is_crawford: bool — True if Crawford game

    Args:
        positions: List of position dicts.
        eval_level: ``"1ply"``, ``"2ply"``, ``"3ply"``, or ``"4ply"``.
        weights: Weight configuration (defaults to production model).
        n_threads: Number of threads for position-level parallelism
            (0 = auto-detect all cores).
        filter_max_moves: Max moves for N-ply move filter.
        filter_threshold: Equity threshold for N-ply move filter.
        jacoby: If True, gammons/backgammons don't count when cube is
            centered (money games only). Ignored for match play positions.
        beaver: If True, opponent can beaver after being doubled
            (money games only). Ignored for match play positions.
        models: Loaded strategies to use and keep (see :class:`BatchModels`);
            give this or ``weights``.
        bearoff_db: The bearoff database, as in :class:`BatchModels` (default
            on, like the analyzer); give it with ``weights``, not ``models``.

    Returns:
        List of :class:`PositionEval`, one per input position.
    """
    n_plies = _check_level(eval_level)
    models = _resolve_models(weights, models, bearoff_db)
    if not positions:
        return []

    # Build (board, cube_value, CubeOwner[, away1, away2, is_crawford]) tuples for C++
    cpp_positions = []
    for p in positions:
        board = list(p["board"])
        cube_value = p.get("cube_value", 1)
        cube_owner = resolve_owner(p.get("cube_owner", "centered"))
        away1 = p.get("away1", 0)
        away2 = p.get("away2", 0)
        is_crawford = p.get("is_crawford", False)
        if away1 > 0 or away2 > 0:
            cpp_positions.append((board, cube_value, cube_owner,
                                  away1, away2, is_crawford))
        else:
            cpp_positions.append((board, cube_value, cube_owner))

    # Jacoby and beaver apply to the money positions only: the native side
    # checks each position's match state (CubeInfo::jacoby_active needs
    # is_money(), and the beaver lives on the money paths), so match rows in
    # the same batch ignore the flags and money rows keep them.

    strategy = (models.base() if n_plies == 1 else
                models.multipy(n_plies, filter_max_moves, filter_threshold))
    raw_results = bgbot_cpp.batch_evaluate_positions(
        cpp_positions, strategy, _auto_threads(n_threads, len(cpp_positions), n_plies),
        jacoby=jacoby, beaver=beaver, **_root_cube_life_kwargs(),
    )

    return [
        PositionEval(
            probs=Probabilities.from_list(list(r["probs"])),
            cubeless_equity=r["cubeless_equity"],
            cubeful_equity=r["cubeful_equity"],
            equity_nd=r["equity_nd"],
            equity_dt=r["equity_dt"],
            equity_dp=r["equity_dp"],
            should_double=bool(r["should_double"]),
            should_take=bool(r["should_take"]),
            optimal_action=r["optimal_action"],
            is_beaver=bool(r.get("is_beaver", False)),
        )
        for r in raw_results
    ]


def batch_post_move_evaluate(
    positions: list[dict],
    eval_level: str = "1ply",
    weights: WeightConfig | WeightConfigPair | None = None,
    *,
    n_threads: int = 0,
    filter_max_moves: int = 5,
    filter_threshold: float = 0.08,
    jacoby: bool = True,
    models: BatchModels | None = None,
    bearoff_db: bool | str | None = None,
) -> list[PostMoveAnalysis]:
    """Evaluate a batch of post-move positions in parallel.

    "Post-move" means the board is from the perspective of the player who
    just moved, right before the opponent rolls.  The NN is evaluated
    directly (no flip/invert), producing post-move probabilities.

    Each position dict must contain:
        board: list[int]   — 26-element post-move board (mover's perspective)
        cube_owner: str    — "centered", "player", or "opponent"

    Optional match play fields:
        cube_value: int    — current cube value (default 1)
        away1: int         — points player needs (0 = money game)
        away2: int         — points opponent needs (0 = money game)
        is_crawford: bool  — True if Crawford game

    Args:
        positions: List of position dicts.
        eval_level: ``"1ply"``, ``"2ply"``, ``"3ply"``, or ``"4ply"``.
        weights: Weight configuration (defaults to production model).
        n_threads: Number of threads for position-level parallelism
            (0 = auto-detect all cores).
        filter_max_moves: Max moves for N-ply move filter.
        filter_threshold: Equity threshold for N-ply move filter.
        jacoby: If True, gammons/backgammons don't count when cube is
            centered (money games only). Ignored for match play positions.
        models: Loaded strategies to use and keep (see :class:`BatchModels`);
            give this or ``weights``.
        bearoff_db: The bearoff database, as in :class:`BatchModels` (default
            on, like the analyzer); give it with ``weights``, not ``models``.

    Returns:
        List of :class:`~bgsage.types.PostMoveAnalysis`, one per input position.
    """
    n_plies = _check_level(eval_level)
    models = _resolve_models(weights, models, bearoff_db)
    if not positions:
        return []

    # Build (board, CubeOwner[, cube_value, away1, away2, is_crawford]) tuples for C++
    cpp_positions = []
    for p in positions:
        board = list(p["board"])
        cube_owner = resolve_owner(p.get("cube_owner", "centered"))
        away1 = p.get("away1", 0)
        away2 = p.get("away2", 0)
        is_crawford = p.get("is_crawford", False)
        cube_value = p.get("cube_value", 1)
        if away1 > 0 or away2 > 0:
            cpp_positions.append((board, cube_owner, cube_value,
                                  away1, away2, is_crawford))
        else:
            cpp_positions.append((board, cube_owner))

    # Jacoby applies to the money positions only; the native side checks
    # each position's match state, so a match row never switches it off for
    # the money rows of the same batch.

    strategy = (models.base() if n_plies == 1 else
                models.multipy(n_plies, filter_max_moves, filter_threshold))
    raw_results = bgbot_cpp.batch_evaluate_post_move(
        cpp_positions, strategy, _auto_threads(n_threads, len(cpp_positions), n_plies),
        jacoby=jacoby, **_root_cube_life_kwargs(),
    )
    level_label = f"{n_plies}-ply"

    return [
        PostMoveAnalysis(
            probs=Probabilities.from_list(list(r["probs"])),
            cubeless_equity=r["cubeless_equity"],
            cubeful_equity=r["cubeful_equity"],
            eval_level=level_label,
        )
        for r in raw_results
    ]


def batch_cube_action(
    positions: list[dict],
    eval_level: str = "1ply",
    weights: WeightConfig | WeightConfigPair | None = None,
    *,
    n_threads: int = 0,
    filter_max_moves: int = 5,
    filter_threshold: float = 0.08,
    jacoby: bool = True,
    beaver: bool = True,
    models: BatchModels | None = None,
    bearoff_db: bool | str | None = None,
) -> list[CubeActionResult]:
    """Evaluate cube decisions for a batch of pre-roll positions in parallel.

    This is a convenience wrapper around :func:`batch_evaluate` that returns
    :class:`~bgsage.types.CubeActionResult` objects (the same type returned by
    ``BgBotAnalyzer.cube_action()``).

    Cube decisions use Janowski interpolation on the N-ply cubeless probs
    (same approach as ``batch_evaluate``), **not** the expensive recursive
    N-ply cubeful search from ``cube_decision_nply``.  This is much faster
    and adequate for most use cases.  In money positions the conversion uses
    the analyzer's 1-ply contact cube life at every depth
    (``ROOT_JANOWSKI_CUBE_LIFE_PLY``): Janowski at the root is the construct
    that value was fitted for.

    Each position dict must contain:
        board: list[int]  — 26-element board from mover's perspective
        cube_value: int   — current cube value (1, 2, 4, ...)
        cube_owner: str   — "centered", "player", or "opponent"

    Optional match play fields:
        away1: int        — points player needs (0 = money game)
        away2: int        — points opponent needs (0 = money game)
        is_crawford: bool — True if Crawford game

    Args:
        positions: List of position dicts.
        eval_level: ``"1ply"``, ``"2ply"``, ``"3ply"``, or ``"4ply"``.
        weights: Weight configuration (defaults to production model).
        n_threads: Number of threads for position-level parallelism
            (0 = auto-detect all cores).
        filter_max_moves: Max moves for N-ply move filter.
        filter_threshold: Equity threshold for N-ply move filter.
        models: Loaded strategies to use and keep (see :class:`BatchModels`);
            give this or ``weights``.
        bearoff_db: The bearoff database, as in :class:`BatchModels` (default
            on, like the analyzer); give it with ``weights``, not ``models``.

    Returns:
        List of :class:`~bgsage.types.CubeActionResult`, one per input position.
    """
    evals = batch_evaluate(
        positions,
        eval_level=eval_level,
        weights=weights,
        n_threads=n_threads,
        filter_max_moves=filter_max_moves,
        filter_threshold=filter_threshold,
        jacoby=jacoby,
        beaver=beaver,
        models=models,
        bearoff_db=bearoff_db,
    )

    level_label = f"{_check_level(eval_level)}-ply"

    return [
        CubeActionResult(
            probs=e.probs,
            cubeless_equity=e.cubeless_equity,
            equity_nd=e.equity_nd,
            equity_dt=e.equity_dt,
            equity_dp=e.equity_dp,
            should_double=e.should_double,
            should_take=e.should_take,
            optimal_equity=(
                e.equity_nd if not e.should_double
                else e.equity_dt if e.should_take
                else e.equity_dp
            ),
            optimal_action=e.optimal_action,
            eval_level=level_label,
            is_beaver=e.is_beaver,
        )
        for e in evals
    ]


def batch_checker_play(
    positions: list[dict],
    eval_level: str = "1ply",
    weights: WeightConfig | WeightConfigPair | None = None,
    *,
    n_threads: int = 0,
    filter_max_moves: int = 5,
    filter_threshold: float = 0.08,
    jacoby: bool = True,
    models: BatchModels | None = None,
    bearoff_db: bool | str | None = None,
) -> list[CheckerPlayResult]:
    """Evaluate checker play for a batch of positions in parallel.

    For each input position (board + dice + cube), generates all legal moves,
    scores them, applies filtering, and optionally re-scores survivors at
    N-ply.  Returns the full ranked move list for each position.

    Each position dict must contain:
        board: list[int]    — 26-element pre-move board (mover's perspective)
        die1: int           — first die value (1-6)
        die2: int           — second die value (1-6)
        cube_value: int     — current cube value (1, 2, 4, ...)
        cube_owner: str     — "centered", "player", or "opponent"

    Optional match play fields:
        away1: int          — points player needs (0 = money game)
        away2: int          — points opponent needs (0 = money game)
        is_crawford: bool   — True if Crawford game

    Args:
        positions: List of position dicts.
        eval_level: ``"1ply"``, ``"2ply"``, ``"3ply"``, or ``"4ply"``.
        weights: Weight configuration (defaults to production model).
        n_threads: Number of threads for position-level parallelism
            (0 = auto-detect all cores).
        filter_max_moves: Max moves for N-ply move filter.
        filter_threshold: Equity threshold for N-ply move filter.
        jacoby: If True, gammons/backgammons don't count when cube is
            centered (money games only). Ignored for match play positions.
        models: Loaded strategies to use and keep (see :class:`BatchModels`);
            give this or ``weights``. The N-ply search is built around the
            same base as the 1-ply scoring, so one set of weights is loaded.
        bearoff_db: The bearoff database, as in :class:`BatchModels` (default
            on, like the analyzer); give it with ``weights``, not ``models``.

    Returns:
        List of :class:`~bgsage.types.CheckerPlayResult`, one per input
        position.  Each contains a ``moves`` list sorted best-first by
        cubeful equity: the list ``BgBotAnalyzer.checker_play`` gives at the
        same level and bearoff database setting.  At the depths in
        ``CUBEFUL_FILTERED_PLIES`` (2-ply) the 1-ply filter's survivors and
        promotions are evaluated at the requested ply and the rest at 1-ply;
        at 3-ply and deeper every move is.
    """
    n_plies = _check_level(eval_level)
    models = _resolve_models(weights, models, bearoff_db)
    if not positions:
        return []

    # Build input dicts with CubeOwner enum for C++
    cpp_inputs = []
    for p in positions:
        d = {
            "board": list(p["board"]),
            "die1": p["die1"],
            "die2": p["die2"],
            "cube_value": p.get("cube_value", 1),
            "cube_owner": resolve_owner(p.get("cube_owner", "centered")),
        }
        away1 = p.get("away1", 0)
        away2 = p.get("away2", 0)
        is_crawford = p.get("is_crawford", False)
        if away1 > 0 or away2 > 0:
            d["away1"] = away1
            d["away2"] = away2
            d["is_crawford"] = is_crawford
        cpp_inputs.append(d)

    strategy_1ply = models.base()

    # Jacoby applies to the money positions only; the native side checks
    # each position's match state, so a match row never switches it off for
    # the money rows of the same batch.

    if n_plies == 1:
        raw_results = bgbot_cpp.batch_checker_play(
            cpp_inputs, strategy_1ply,
            filter_max_moves, filter_threshold, n_threads,
            jacoby=jacoby,
            contact_x_1ply=_contact_cube_life(1),
            contact_x_pip_slope_1ply=_contact_cube_slope(1),
        )
    else:
        raw_results = bgbot_cpp.batch_checker_play(
            cpp_inputs, strategy_1ply,
            models.multipy(n_plies, filter_max_moves, filter_threshold),
            filter_max_moves, filter_threshold, n_threads,
            jacoby=jacoby,
            contact_x_1ply=_contact_cube_life(1),
            contact_x_pip_slope_1ply=_contact_cube_slope(1),
            contact_x_nply=_contact_cube_life(n_plies),
            contact_x_pip_slope_nply=_contact_cube_slope(n_plies),
            cubeful_filtered_plies=sorted(CUBEFUL_FILTERED_PLIES),
        )
    level_label = f"{n_plies}-ply"

    results = []
    for i, raw in enumerate(raw_results):
        p = positions[i]
        moves = [
            MoveAnalysis(
                board=list(m["board"]),
                equity=m["equity"],
                cubeless_equity=m["cubeless_equity"],
                probs=Probabilities.from_list(list(m["probs"])),
                equity_diff=m["equity_diff"],
                eval_level=m["eval_level"],
            )
            for m in raw["moves"]
        ]
        results.append(CheckerPlayResult(
            moves=moves,
            board=list(p["board"]),
            die1=p["die1"],
            die2=p["die2"],
            eval_level=level_label,
        ))
    return results
