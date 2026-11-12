# SPDX-License-Identifier: MPL-2.0
"""Public checker-play thread budgets with real native evaluation and pools."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from unittest.mock import patch

import pytest

from bgsage import BgBotAnalyzer, STARTING_BOARD


@pytest.fixture(autouse=True)
def controlled_thread_defaults(monkeypatch):
    # Keep expectations independent of the CI host's CPU count/environment.
    monkeypatch.delenv("BGBOT_MULTIPLY_THREADS", raising=False)
    monkeypatch.setattr("bgsage.analyzer.os.cpu_count", lambda: 8)


def evaluate(level, threads, *, cubeful=True, match_play=False):
    limits = []

    def observe(*args, **kwargs):
        limits.append(kwargs.get("max_workers", args[0] if args else None))
        return ThreadPoolExecutor(*args, **kwargs)

    analyzer = BgBotAnalyzer(eval_level=level, parallel_threads=threads, cubeful=cubeful)
    context = {"away1": 5, "away2": 3} if match_play else {}
    with patch("bgsage.analyzer.ThreadPoolExecutor", side_effect=observe):
        result = analyzer.checker_play(list(STARTING_BOARD), 3, 1, beaver=False, **context)
    assert result.moves, "The real checker evaluation must produce candidates"
    return limits, asdict(result)


@pytest.mark.parametrize("threads", [1, 2, 4])
@pytest.mark.parametrize("match_play", [False, True], ids=["money", "match"])
def test_one_ply_honors_explicit_thread_limit(threads, match_play):
    limits, _ = evaluate("1ply", threads, match_play=match_play)
    assert limits == [threads]


@pytest.mark.parametrize("environment,expected", [(None, 8), ("", 8), ("0", 8),
                                                 ("-1", 8), ("invalid", 8),
                                                 ("1", 1), ("2", 2), ("4", 4)])
def test_one_ply_automatic_budget_uses_existing_default_policy(monkeypatch, environment, expected):
    if environment is not None:
        monkeypatch.setenv("BGBOT_MULTIPLY_THREADS", environment)
    limits, _ = evaluate("1ply", 0)
    assert limits == [expected]


def test_explicit_one_ply_budget_takes_precedence_over_environment(monkeypatch):
    monkeypatch.setenv("BGBOT_MULTIPLY_THREADS", "4")
    limits, _ = evaluate("1ply", 2)
    assert limits == [2]


@pytest.mark.parametrize("threads", [2, 4])
def test_two_ply_explicit_budgets_are_preserved(threads):
    limits, _ = evaluate("2ply", threads)
    assert limits == [threads]


@pytest.mark.parametrize("threads", [1, 2, 4])
def test_cubeless_one_ply_does_not_add_a_pool(threads):
    limits, _ = evaluate("1ply", threads, cubeful=False)
    assert limits == []


@pytest.mark.parametrize("match_play", [False, True], ids=["money", "match"])
def test_one_ply_results_do_not_depend_on_thread_budget(match_play):
    outputs = [evaluate("1ply", n, match_play=match_play)[1] for n in (1, 2, 4)]
    assert outputs[0] == outputs[1] == outputs[2]
