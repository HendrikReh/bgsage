# SPDX-License-Identifier: MPL-2.0
"""The batch functions load each model once and can keep it loaded.

Every batch call used to build its strategies from the weight files: about
15 ms for Stage 11, paid even by an empty batch, and twice by N-ply checker
play, whose N-ply strategy loaded a second copy of the base. A BatchModels
now holds the loaded strategies: the batch functions build one per call
unless given one, the N-ply strategy is built around the same base
(``create_multipy_from_base``), and an empty batch loads nothing.

These tests hold the results to strategies built the old way (one
create_strategy / create_multipy per call) and count the factory calls.
"""

import os
import random
import sys
import threading
from dataclasses import asdict
from unittest.mock import patch

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "build"), os.path.join(_ROOT, "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bgbot_cpp  # noqa: E402

from bgsage import (  # noqa: E402
    BatchModels, STARTING_BOARD, batch_checker_play, batch_cube_action,
    batch_evaluate, batch_post_move_evaluate, flip_board, possible_moves,
)
from bgsage.analyzer import CUBEFUL_FILTERED_PLIES, _contact_cube_life, _contact_cube_slope  # noqa: E402
from bgsage.board import check_game_over  # noqa: E402
from bgsage.weights import WeightConfig, default_weights  # noqa: E402

CUBES = [
    dict(cube_value=1, cube_owner="centered"),
    dict(cube_value=2, cube_owner="opponent"),
    dict(cube_value=1, cube_owner="centered", away1=5, away2=3),
]


def _positions(n, seed):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        board = list(STARTING_BOARD)
        for _ in range(rng.randint(4, 40)):
            moves = possible_moves(board, rng.randint(1, 6), rng.randint(1, 6))
            if moves:
                board = list(rng.choice(moves))
            if check_game_over(board):
                break
            board = list(flip_board(board))
        else:
            d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
            if len(possible_moves(board, d1, d2)) >= 2:
                out.append(dict(board=board, die1=d1, die2=d2,
                                **CUBES[len(out) % len(CUBES)]))
    return out


POSITIONS = _positions(12, seed=5)
PRE_ROLL = [{k: v for k, v in p.items() if k not in ("die1", "die2")} for p in POSITIONS]

MODELS = {"stage11": default_weights(), "stage5": WeightConfig.from_model("stage5")}


def _rows(results):
    return [asdict(r) for r in results]


# ---------------------------------------------------------------------------
# The old construction, one strategy set per call, for reference
# ---------------------------------------------------------------------------


def _old_base(w):
    if w.strategy_type == "5nn":
        return bgbot_cpp.GamePlanStrategy(*w.weight_args)
    return bgbot_cpp.create_strategy(w.strategy_type, w.weight_paths_list, w.hidden_sizes_list)


def _old_multipy(w, n_plies):
    if w.strategy_type == "5nn":
        return bgbot_cpp.create_multipy_5nn(
            *w.weight_args, n_plies=n_plies, filter_max_moves=5, filter_threshold=0.08,
            parallel_evaluate=False, parallel_threads=0)
    return bgbot_cpp.create_multipy(
        w.strategy_type, w.weight_paths_list, w.hidden_sizes_list, n_plies=n_plies,
        filter_max_moves=5, filter_threshold=0.08,
        parallel_evaluate=False, parallel_threads=0)


def _cpp_pre_roll(rows):
    return [(p["board"], p["cube_value"], bgbot_cpp.CubeOwner.CENTERED
             if p["cube_owner"] == "centered" else bgbot_cpp.CubeOwner.OPPONENT,
             *((p["away1"], p["away2"], False) if "away1" in p else ()))
            for p in rows]


@pytest.mark.parametrize("model", list(MODELS))
@pytest.mark.parametrize("level", ["1ply", "2ply"])
def test_results_equal_the_old_construction(model, level):
    w = MODELS[model]
    n = int(level[0])
    strategy = _old_base(w) if n == 1 else _old_multipy(w, n)
    kw = dict(contact_x=_contact_cube_life(1), contact_x_pip_slope=_contact_cube_slope(1)) \
        if n == 1 else {}
    old = bgbot_cpp.batch_evaluate_positions(_cpp_pre_roll(PRE_ROLL), strategy, 2, **kw)
    new = batch_evaluate(PRE_ROLL, eval_level=level, weights=w, n_threads=2)
    for o, r in zip(old, new):
        assert r.cubeful_equity == o["cubeful_equity"]
        assert r.equity_nd == o["equity_nd"] and r.equity_dt == o["equity_dt"]
        assert list(r.probs.to_list()) == pytest.approx(list(o["probs"]), abs=0)

    inputs = [dict(p, cube_owner=bgbot_cpp.CubeOwner.CENTERED
                   if p["cube_owner"] == "centered" else bgbot_cpp.CubeOwner.OPPONENT)
              for p in POSITIONS]
    if n == 1:
        old_cp = bgbot_cpp.batch_checker_play(
            inputs, _old_base(w), 5, 0.08, 2,
            contact_x_1ply=_contact_cube_life(1),
            contact_x_pip_slope_1ply=_contact_cube_slope(1))
    else:
        old_cp = bgbot_cpp.batch_checker_play(
            inputs, _old_base(w), _old_multipy(w, n), 5, 0.08, 2,
            contact_x_1ply=_contact_cube_life(1),
            contact_x_pip_slope_1ply=_contact_cube_slope(1),
            contact_x_nply=_contact_cube_life(n),
            contact_x_pip_slope_nply=_contact_cube_slope(n),
            cubeful_filtered_plies=sorted(CUBEFUL_FILTERED_PLIES))
    new_cp = batch_checker_play(POSITIONS, eval_level=level, weights=w, n_threads=2)
    for o, r in zip(old_cp, new_cp):
        assert [(tuple(m["board"]), m["equity"], m["eval_level"]) for m in o["moves"]] == \
            [(tuple(m.board), m.equity, m.eval_level) for m in r.moves]


@pytest.mark.parametrize("level", ["1ply", "2ply", "3ply"])
def test_reused_models_give_the_same_results(level):
    models = BatchModels()
    for _ in range(2):
        assert _rows(batch_checker_play(POSITIONS[:6], eval_level=level, models=models)) == \
            _rows(batch_checker_play(POSITIONS[:6], eval_level=level))
        assert _rows(batch_evaluate(PRE_ROLL, eval_level=level, models=models)) == \
            _rows(batch_evaluate(PRE_ROLL, eval_level=level))
        post = [dict(p, board=flip_board(p["board"])) for p in PRE_ROLL]
        assert _rows(batch_post_move_evaluate(post, eval_level=level, models=models)) == \
            _rows(batch_post_move_evaluate(post, eval_level=level))
        assert _rows(batch_cube_action(PRE_ROLL, eval_level=level, models=models)) == \
            _rows(batch_cube_action(PRE_ROLL, eval_level=level))


# ---------------------------------------------------------------------------
# Factory calls
# ---------------------------------------------------------------------------


FACTORIES = ("create_strategy", "create_multipy", "create_multipy_5nn",
             "create_multipy_from_base", "GamePlanStrategy")


class _Count:
    def __enter__(self):
        self._patches = [patch.object(bgbot_cpp, name, wraps=getattr(bgbot_cpp, name))
                         for name in FACTORIES]
        self.mocks = {name: p.start() for name, p in zip(FACTORIES, self._patches)}
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()

    def calls(self):
        return {name: m.call_count for name, m in self.mocks.items() if m.call_count}


@pytest.mark.parametrize("fn", [batch_evaluate, batch_post_move_evaluate,
                                batch_cube_action, batch_checker_play])
@pytest.mark.parametrize("level", ["1ply", "2ply"])
def test_an_empty_batch_loads_nothing(fn, level):
    with _Count() as c:
        assert fn([], eval_level=level) == []
    assert c.calls() == {}


@pytest.mark.parametrize("fn", [batch_evaluate, batch_post_move_evaluate,
                                batch_cube_action, batch_checker_play])
def test_an_invalid_level_raises_even_when_empty(fn):
    with pytest.raises(ValueError):
        fn([], eval_level="5ply")
    with pytest.raises(ValueError):
        fn(POSITIONS[:1], eval_level="0ply", models=BatchModels())


def test_n_ply_checker_play_loads_one_base():
    with _Count() as c:
        batch_checker_play(POSITIONS[:2], eval_level="2ply")
    assert c.calls() == {"create_strategy": 1, "create_multipy_from_base": 1}


def test_reused_models_load_once():
    models = BatchModels()
    with _Count() as c:
        for level in ("1ply", "2ply", "3ply", "2ply", "1ply"):
            batch_checker_play(POSITIONS[:2], eval_level=level, models=models)
            batch_evaluate(PRE_ROLL[:2], eval_level=level, models=models)
    assert c.calls() == {"create_strategy": 1, "create_multipy_from_base": 2}


def test_weights_and_models_together_must_agree():
    models = BatchModels()
    assert batch_evaluate(PRE_ROLL[:1], weights=models.weights, models=models)
    with pytest.raises(ValueError):
        batch_evaluate(PRE_ROLL[:1], weights=MODELS["stage5"], models=models)


def test_concurrent_calls_can_share_models():
    models = BatchModels()
    expected = _rows(batch_checker_play(POSITIONS, eval_level="2ply", n_threads=2))
    results, errors = [], []

    def run():
        try:
            results.append(_rows(batch_checker_play(
                POSITIONS, eval_level="2ply", n_threads=2, models=models)))
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert results == [expected] * 4
