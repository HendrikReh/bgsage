# SPDX-License-Identifier: MPL-2.0
"""The analyzer loads each model once on its checker side and shares the
bearoff database; empty batches load nothing.

A 3-ply BgBotAnalyzer loaded Stage 11 three times: for its 1-ply strategy,
again inside create_multipy for the N-ply strategy, and once more for the
cube strategy on its first cube action. Every analyzer also read the bearoff
database from disk. Now the N-ply strategy is built around the 1-ply
strategy's base (create_multipy_from_base), the database is loaded once per
process, and BatchModels loads it only when a batch has positions. The cube
strategy keeps its own base, so its evaluator configuration stays separate.
"""

import os
import sys
from dataclasses import asdict
from unittest.mock import patch

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "build"), os.path.join(_ROOT, "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bgbot_cpp  # noqa: E402

import bgsage.analyzer as A  # noqa: E402
from bgsage import (  # noqa: E402
    BatchModels, BgBotAnalyzer, STARTING_BOARD, batch_checker_play, batch_evaluate,
    batch_post_move_evaluate,
)
from bgsage.weights import WeightConfig, bearoff_db_path  # noqa: E402

FACTORIES = ("create_strategy", "create_multipy", "create_multipy_from_base",
             "create_multipy_5nn", "GamePlanStrategy", "BearoffDB")
SNAKE = [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 2, 3, 3, 2, 2, 1, -7, -7, 0]
BEAROFF = [0, 3, 3, 2, 3, 2, 2] + [0] * 12 + [-2, -3, -2, -3, -2, -2] + [0]
CONTACT = [0, 0, 0, 0, 2, 2, 4, -2, 2, 0, 2, 0, -2, 3, -1, 0, -1, -2, 0, -2, -3, -2, 0, 0, 0, 0]


class _Count:
    def __enter__(self):
        self._patches = [patch.object(bgbot_cpp, n, wraps=getattr(bgbot_cpp, n)) for n in FACTORIES]
        self.mocks = {n: p.start() for n, p in zip(FACTORIES, self._patches)}
        A.load_bearoff_db.cache_clear()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()

    def take(self):
        out = {n: m.call_count for n, m in self.mocks.items() if m.call_count}
        for m in self.mocks.values():
            m.reset_mock()
        return out


def test_loads_at_each_stage_of_a_three_ply_analyzer():
    with _Count() as c:
        a = BgBotAnalyzer(eval_level="3ply", parallel_threads=2)
        assert c.take() == {"create_strategy": 1, "create_multipy_from_base": 1, "BearoffDB": 1}
        a.checker_play(STARTING_BOARD, 3, 1)
        assert c.take() == {}
        a.cube_action(STARTING_BOARD)
        assert c.take() == {"create_strategy": 1}     # the cube strategy, once
        a.cube_action(CONTACT)
        a.checker_play(CONTACT, 6, 2)
        assert c.take() == {}
        BgBotAnalyzer(eval_level="2ply", parallel_threads=2)
        assert c.take() == {"create_strategy": 1, "create_multipy_from_base": 1}  # no DB reload


@pytest.mark.parametrize("fn", [batch_evaluate, batch_post_move_evaluate, batch_checker_play])
def test_an_empty_batch_loads_no_model_and_no_database(fn):
    with _Count() as c:
        assert fn([], eval_level="2ply") == []
        assert c.take() == {}
    with pytest.raises(ValueError):
        fn([], eval_level="5ply")
    with pytest.raises(ValueError):
        fn([], models=BatchModels(), bearoff_db=False)


def _old_nply(analyzer):
    """Rebuild the analyzer's N-ply strategy the old way: its own base."""
    inner = analyzer._analyzer._inner
    w = inner._weights
    s = bgbot_cpp.create_multipy(w.strategy_type, w.weight_paths_list, w.hidden_sizes_list,
                                 n_plies=inner._n_plies, parallel_evaluate=True,
                                 parallel_threads=inner._parallel_threads)
    if inner._bearoff_db is not None:
        bgbot_cpp.multipy_set_bearoff_db(s, inner._bearoff_db)
    inner._strategy_nply = s


def _results(a):
    out = []
    for board, d in ((CONTACT, (6, 2)), (SNAKE, (6, 5)), (BEAROFF, (4, 2))):
        out.append([asdict(m) for m in a.checker_play(board, *d).moves])
        out.append(asdict(a.cube_action(board)))
        out.append(asdict(a.post_move_analytics(board, cube_owner="centered")))
    return out


@pytest.mark.parametrize("model,bearoff_db", [
    ("stage11", True), ("stage11", False), ("stage5", True), ("stage11", "path"),
])
@pytest.mark.parametrize("level", ["2ply", "3ply"])
def test_results_equal_the_old_construction(model, bearoff_db, level):
    w = WeightConfig.from_model(model) if model != "stage11" else None
    db = bearoff_db_path() if bearoff_db == "path" else bearoff_db
    new = BgBotAnalyzer(weights=w, eval_level=level, parallel_threads=2, bearoff_db=db)
    old = BgBotAnalyzer(weights=w, eval_level=level, parallel_threads=2, bearoff_db=db)
    _old_nply(old)
    assert _results(new) == _results(old)
