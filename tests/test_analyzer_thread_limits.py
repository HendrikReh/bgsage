# SPDX-License-Identifier: MPL-2.0
"""Public checker-play thread budgets with real native evaluation and pools."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from unittest.mock import patch

import pytest

from neongammon import BgBotAnalyzer, STARTING_BOARD


@pytest.fixture(autouse=True)
def controlled_thread_defaults(monkeypatch):
    # Keep expectations independent of the CI host's CPU count/environment.
    monkeypatch.delenv("BGBOT_MULTIPLY_THREADS", raising=False)
    monkeypatch.setattr("neongammon.analyzer.os.cpu_count", lambda: 8)


def evaluate(level, threads, *, cubeful=True, match_play=False):
    limits = []

    def observe(*args, **kwargs):
        limits.append(kwargs.get("max_workers", args[0] if args else None))
        return ThreadPoolExecutor(*args, **kwargs)

    analyzer = BgBotAnalyzer(eval_level=level, parallel_threads=threads, cubeful=cubeful)
    context = {"away1": 5, "away2": 3} if match_play else {}
    with patch("neongammon.analyzer.ThreadPoolExecutor", side_effect=observe):
        result = analyzer.checker_play(list(STARTING_BOARD), 3, 1, beaver=False, **context)
    assert result.moves, "The real checker evaluation must produce candidates"
    return limits, asdict(result)


@pytest.mark.parametrize("threads", [1, 2, 4])
@pytest.mark.parametrize("match_play", [False, True], ids=["money", "match"])
def test_one_ply_honors_explicit_thread_limit(threads, match_play):
    limits, _ = evaluate("1ply", threads, match_play=match_play)
    assert limits == []


@pytest.mark.parametrize("level", ["2ply", "3ply"])
@pytest.mark.parametrize("threads", [1, 2, 4])
def test_n_ply_explicit_budgets_reach_the_native_routine(level, threads):
    # Cube-aware N-ply checker play spreads its cube-tree calls natively
    # (bgbot_cpp.checker_play_nply) over exactly the analyzer's budget.
    import bgbot_cpp
    with patch.object(bgbot_cpp, "checker_play_nply", wraps=bgbot_cpp.checker_play_nply) as native:
        limits, _ = evaluate(level, threads)
    assert limits == []
    assert native.call_args.kwargs["n_threads"] == threads


@pytest.mark.parametrize("threads", [1, 2, 4])
def test_cubeless_one_ply_does_not_add_a_pool(threads):
    limits, _ = evaluate("1ply", threads, cubeful=False)
    assert limits == []


@pytest.mark.parametrize("match_play", [False, True], ids=["money", "match"])
def test_one_ply_results_do_not_depend_on_thread_budget(match_play):
    outputs = [evaluate("1ply", n, match_play=match_play)[1] for n in (1, 2, 4)]
    assert outputs[0] == outputs[1] == outputs[2]


# ---------------------------------------------------------------------------
# An explicit budget of one runs inline (it used to be raised to two).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", ["2ply", "3ply"])
@pytest.mark.parametrize("match_play", [False, True], ids=["money", "match"])
def test_n_ply_budget_of_one_runs_inline(level, match_play):
    limits, _ = evaluate(level, 1, match_play=match_play)
    assert limits == []
    assert BgBotAnalyzer(eval_level=level, parallel_threads=1)._analyzer._inner._parallel_threads == 1


@pytest.mark.parametrize("level", ["2ply", "3ply"])
@pytest.mark.parametrize("match_play", [False, True], ids=["money", "match"])
def test_n_ply_results_do_not_depend_on_thread_budget(level, match_play):
    outputs = [evaluate(level, n, match_play=match_play)[1] for n in (1, 2, 4, 0)]
    assert all(o == outputs[0] for o in outputs)


def test_n_ply_cube_action_does_not_depend_on_thread_budget():
    board = [0, 0, 0, 0, 2, 2, 4, -2, 2, 0, 2, 0, -2, 3, -1, 0, -1, -2, 0, -2, -3, -2, 0, 0, 0, 0]
    out = [asdict(BgBotAnalyzer(eval_level="3ply", parallel_threads=n).cube_action(board))
           for n in (1, 2, 4)]
    assert out[0] == out[1] == out[2]
