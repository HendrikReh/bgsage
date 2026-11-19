#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""Fit the money contact cube life to XG's cube decisions.

Fits ``contact_x + slope * (average pips - 120)`` (``CubeInfo.contact_x`` /
``contact_x_pip_slope``) to XG World Class's verdicts on the money
benchmark's cube positions, the set ``export_money_cube_xg.py`` writes and
XG Batch-Analyzes.

1. For every point of a (contact_x, slope) grid, play every cube decision
   with the engine at each level, the analyzer's tables set to that point:
   every position the cube tree converts, root and leaves, takes the cube
   life of its own pip count, exactly as in production.
2. Fit (contact_x, slope) by grid search, minimising the cube error against
   XG on half the seeds and scoring the other half, both ways round, plus the
   flat model (slope 0) with the same procedure. Reported against XG and
   against the Sage reference, with the paired out-of-fold difference.

Usage::

    python scripts/fit_contact_cube_life.py --xg-dir data/money_cube_xg
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_SCRIPT_DIR, _PROJECT_ROOT / "build", _PROJECT_ROOT / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import bgbot_cpp  # noqa: E402
import bgsage.analyzer as A  # noqa: E402
from bgsage import BgBotAnalyzer  # noqa: E402

import benchmark_money as bm  # noqa: E402
import export_money_cube_xg as X  # noqa: E402

A_GRID = [round(0.70 + 0.0125 * i, 4) for i in range(13)]       # 0.70 .. 0.85
B_GRID = [round(0.00025 * i, 5) for i in range(-4, 13)]        # -0.001 .. 0.003


def sweep(rows: list[dict], levels: list[str], threads: int) -> dict:
    """picks[level][(a, b)] = [(should_double, should_take) per row]."""
    local = threading.local()
    saved = dict(A.MONEY_CONTACT_CUBE_LIFE), dict(A.MONEY_CONTACT_CUBE_LIFE_PIP_SLOPE)
    picks: dict = {}

    def run(level, idx):
        def one(i):
            if getattr(local, "level", None) != level:
                local.level = level
                local.an = BgBotAnalyzer(eval_level=level, cubeful=True, parallel_threads=1)
            r = rows[i]
            c = local.an.cube_action(r["board"], cube_value=r["cube_value"],
                                     cube_owner=r["cube_owner"], jacoby=True, beaver=True)
            return bool(c.should_double), bool(c.should_take)
        with ThreadPoolExecutor(threads) as ex:
            return list(ex.map(one, idx, chunksize=32))

    contact = [i for i, r in enumerate(rows) if not r["race"]]
    race = [i for i, r in enumerate(rows) if r["race"]]
    try:
        for level in levels:
            n = int(level[0])
            race_picks = dict(zip(race, run(level, race)))  # contact_x leaves races alone
            picks[level] = {}
            for a, b in itertools.product(A_GRID, B_GRID):
                A.MONEY_CONTACT_CUBE_LIFE[n] = a
                A.MONEY_CONTACT_CUBE_LIFE_PIP_SLOPE[n] = b
                got = {**race_picks, **dict(zip(contact, run(level, contact)))}
                picks[level][(a, b)] = [got[i] for i in range(len(rows))]
            print(f"swept {level}: {len(A_GRID) * len(B_GRID)} grid points", flush=True)
    finally:
        for table, old in zip((A.MONEY_CONTACT_CUBE_LIFE, A.MONEY_CONTACT_CUBE_LIFE_PIP_SLOPE),
                              saved):
            table.clear()
            table.update(old)
    return picks


def per_position(rows, idx, picks_of, ref):
    """Sub-decision errors per position; picks_of(i) gives that row's picks."""
    return [X.errors(rows[i][ref], rows[i]["has_double"], rows[i]["has_take"], *picks_of(i))
            for i in idx]


def summarise(errs):
    flat = [v for e in errs for v in e]
    return (sum(flat) / len(flat) * 500, sum(v > bm.BLUNDER_THRESHOLD for v in flat),
            [sum(e) for e in errs])


def paired(a, b, k):
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    return m * k, math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1) / n) * k


def fit(rows, idx, grid, slopes):
    def total(a, b):
        return sum(sum(e) for e in per_position(rows, idx, lambda i: grid[(a, b)][i], "xg"))
    return min((total(a, b), a, b) for a in A_GRID for b in slopes)[1:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--xg-dir", type=Path, default=X.DEFAULT_OUT_DIR)
    ap.add_argument("--levels", default="1ply,2ply")
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    levels = args.levels.split(",")

    xg_rows, unanalysed = X.harvest(args.xg_dir, args.xg_dir)
    by_key = {e["key"]: e for e in X.load_cube_entries()}
    rows = []
    for r in xg_rows:
        e = by_key[r["key"]]
        rows.append({**{k: r[k] for k in ("board", "cube_value", "cube_owner",
                                          "has_double", "has_take")},
                     "seed": e["seed"], "race": bool(bgbot_cpp.is_race(r["board"])),
                     "xg": r["xg"], "sage": X.sage_ref(e)})
    n_dec = sum(r["has_double"] + r["has_take"] for r in rows)
    print(f"{len(rows)} positions ({unanalysed} unanalysed by XG), {n_dec} cube decisions")
    picks = sweep(rows, levels, args.threads)

    every = list(range(len(rows)))
    halves = {h: [i for i in every if rows[i]["seed"] % 2 == h] for h in (0, 1)}
    for level in levels:
        grid = picks[level]
        print(f"\n{level}")
        oof_err = {}
        for name, slopes in (("flat", [0.0]), ("pip", B_GRID)):
            fits = {h: fit(rows, halves[h], grid, slopes) for h in (0, 1)}
            all_fit = fit(rows, every, grid, slopes)
            print(f"  {name:4s} fitted on even / odd / all seeds: " + " / ".join(
                f"{a:.4f} + {b:.5f}" for a, b in (fits[0], fits[1], all_fit)))
            # each half scored by the model fitted on the other half
            oof = lambda i, fits=fits: grid[fits[1 - rows[i]["seed"] % 2]][i]  # noqa: E731
            for ref in ("xg", "sage"):
                p, b, e = summarise(per_position(rows, every, oof, ref))
                oof_err[(name, ref)] = e
                print(f"    out of fold vs {ref:4s}: cube PR {p:.2f} ({b} blunders)")
        for ref in ("xg", "sage"):
            m, se = paired(oof_err[("flat", ref)], oof_err[("pip", ref)], 500 * len(rows) / n_dec)
            print(f"  pip - flat, out of fold, vs {ref:4s}: {m:+.2f} +- {se:.2f}")


if __name__ == "__main__":
    main()
