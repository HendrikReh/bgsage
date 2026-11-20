# SPDX-License-Identifier: MPL-2.0
"""The bearoff database is part of the evaluation cache identity.

BearoffStrategy forwarded its base's eval_identity, which keys the cubeful
engine's per-thread cache (and the rollout's shared cache derived from it).
So on one base strategy an evaluation with the database and one without
shared cache entries, and whichever ran first decided both: in the bearoff
race below, 2-ply equity +0.3049 without the database and +0.1085 with it,
and on a warm thread the second call returned the first call's value. The
legacy MultiPlyStrategy cache was salted by the base pointer alone, with the
same effect for two N-ply strategies around one base.

Now the database's data has a process-unique identity (new on every load or
generate), the wrapper mixes it into the base's, a wrapper over a wrapper of
the same database shares it, and the legacy key folds it in too.
"""

import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "build"), os.path.join(_ROOT, "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bgbot_cpp  # noqa: E402

from bgsage.weights import bearoff_db_path, default_weights  # noqa: E402

pytestmark = pytest.mark.skipif(bearoff_db_path() is None, reason="no bearoff database")

# Player still bearing in (6- and 4-point), opponent with one checker back.
BOARD = [0] * 26
BOARD[4], BOARD[6], BOARD[19], BOARD[22] = 3, 5, -6, -1


def _db():
    db = bgbot_cpp.BearoffDB()
    assert db.load(bearoff_db_path())
    return db


def _base():
    w = default_weights()
    return bgbot_cpp.create_strategy(w.strategy_type, w.weight_paths_list, w.hidden_sizes_list)


def _tree(strategy, db, plies=2):
    r = bgbot_cpp.cubeful_probs_and_equity_nply(
        BOARD, bgbot_cpp.CubeOwner.CENTERED, strategy, plies, bearoff_db=db)
    return [r["equity"], *r["probs"]]


def _fresh(db, plies=2):
    return _tree(_base(), db, plies)


@pytest.mark.parametrize("plies", [2, 3])
def test_with_and_without_the_database_in_either_order(plies):
    db = _db()
    ref = {"nn": _fresh(None, plies), "db": _fresh(db, plies)}
    assert ref["nn"] != ref["db"]          # the database does change this race
    for order in (("nn", "db"), ("db", "nn")):
        base = _base()
        for name in order:
            assert _tree(base, db if name == "db" else None, plies) == ref[name], order


def test_distinct_database_instances_and_repeated_use():
    a, b = _db(), _db()
    ref = _fresh(a)
    base = _base()
    assert _tree(base, a) == ref
    assert _tree(base, b) == ref           # same data, own identity: recomputed alike
    assert _tree(base, a) == ref           # repeated use of the same database
    assert _tree(base, None) == _fresh(None)


def test_nested_wrappers_over_the_same_database():
    # The analyzer hands the engine an already wrapped strategy and the
    # binding wraps it again; both layers must evaluate (and cache) as one.
    db = _db()
    ref = _fresh(db)
    wrapped = bgbot_cpp.BearoffStrategy(_base(), db)
    assert _tree(wrapped, None) == ref
    assert _tree(wrapped, db) == ref
    assert _tree(bgbot_cpp.BearoffStrategy(wrapped, db), None) == ref


def test_legacy_multiply_cache_tells_the_database_apart():
    # Two full-depth (legacy recursion) N-ply strategies around ONE base,
    # one with the database attached: their cache keys used to coincide.
    # The player is home, one opponent checker still outside: the cached
    # nodes are not bearoff positions, the leaves after the opponent brings
    # that checker home are.
    db = _db()
    pre = [0] * 26
    pre[6], pre[5], pre[4], pre[3] = 4, 4, 3, 3
    pre[19], pre[20], pre[21], pre[16] = -5, -5, -4, -1
    post = [list(c) for c in bgbot_cpp.possible_moves(pre, 5, 2)]

    def strategies(base):
        plain = bgbot_cpp.create_multipy_from_base(base, n_plies=2, full_depth_opponent=True)
        with_db = bgbot_cpp.create_multipy_from_base(base, n_plies=2, full_depth_opponent=True)
        bgbot_cpp.multipy_set_bearoff_db(with_db, db)
        return plain, with_db

    def probs(strategy):
        return [list(strategy.evaluate_board(c, pre)["probs"]) for c in post]

    plain, _ = strategies(_base())
    plain.clear_cache()
    ref_plain = probs(plain)
    _, with_db = strategies(_base())
    with_db.clear_cache()
    ref_db = probs(with_db)
    assert ref_plain != ref_db
    for first in ("db", "plain"):
        plain, with_db = strategies(_base())
        plain.clear_cache()
        order = (with_db, plain) if first == "db" else (plain, with_db)
        got = [probs(s) for s in order]
        if first == "db":
            got.reverse()
        assert got == [ref_plain, ref_db], first
