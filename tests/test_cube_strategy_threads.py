# SPDX-License-Identifier: MPL-2.0
"""One analyzer's cached cube strategy, called from many threads at once,
must return exactly what a fresh analyzer per call returns.

BgBotAnalyzer loads its cube strategy once and reuses it (PR #3). The N-ply
cube bindings release the GIL, and the cubeful engine's per-thread cache can
now hit across calls, because the strategy's identity no longer changes per
call. This pins both: concurrent use of the shared strategy, and cache hits
across calls, cube states and match scores, are bit-identical to the
fresh-strategy behaviour.
"""
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import astuple

import pytest

from neongammon import BgBotAnalyzer, STARTING_BOARD, check_game_over, flip_board, possible_moves

CUBE_STATES = [
    dict(cube_value=1, cube_owner="centered"),
    dict(cube_value=2, cube_owner="player"),
    dict(cube_value=1, cube_owner="centered", away1=5, away2=3),
    dict(cube_value=2, cube_owner="player", away1=7, away2=4),
]


def _positions(n, seed=7):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b = list(STARTING_BOARD)
        for _ in range(rng.randint(4, 60)):
            moves = possible_moves(b, rng.randint(1, 6), rng.randint(1, 6))
            if moves:
                b = list(rng.choice(moves))
            if check_game_over(b):
                break
            b = flip_board(b)
        if not check_game_over(b):
            out.append(b)
    return out


def _key(r):
    return (r.equity_nd, r.equity_dt, r.equity_dp, r.should_double, r.should_take,
            astuple(r.probs))


@pytest.mark.parametrize("level,parallel_threads", [
    ("1ply", 1), ("2ply", 1), ("3ply", 1), ("2ply", 0),
])
def test_shared_cube_strategy_matches_fresh_analyzers(level, parallel_threads):
    positions = _positions(10)
    jobs = [(i, s) for i in range(len(positions)) for s in range(len(CUBE_STATES))]
    want = {j: _key(BgBotAnalyzer(eval_level=level, parallel_threads=1)
                    .cube_action(positions[j[0]], **CUBE_STATES[j[1]]))
            for j in jobs}

    shared = BgBotAnalyzer(eval_level=level, parallel_threads=parallel_threads)
    work = jobs * 2  # repeats exercise cache hits across calls
    random.Random(level).shuffle(work)
    with ThreadPoolExecutor(max_workers=8) as pool:
        got = list(pool.map(
            lambda j: (j, _key(shared.cube_action(positions[j[0]], **CUBE_STATES[j[1]]))),
            work))
    mismatches = [j for j, r in got if r != want[j]]
    assert not mismatches, f"{len(mismatches)} of {len(got)} calls differ, e.g. {mismatches[:3]}"
