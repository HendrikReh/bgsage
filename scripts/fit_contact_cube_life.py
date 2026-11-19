#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""Fit the money contact cube life to XG's cube decisions.

Fits ``contact_x + slope * (average pips - 120)`` (``CubeInfo.contact_x`` /
``contact_x_pip_slope``) to XG World Class's verdicts on the money
benchmark's cube positions, the set ``export_money_cube_xg.py`` writes and
XG Batch-Analyzes.

1. For every contact position and every flat cube life on a 0.025 grid
   (0.55-1.00), record Sage's double / take picks at each level. A cube
   life given by a position's own pips is the flat value at that position's
   pips, and at 1-ply a cube decision is a single Janowski conversion of the
   position, so this table scores any pip model exactly at 1-ply. At 2-ply
   it applies the root's value to the whole tree, an approximation of the
   engine (which takes each leaf's own pips).
2. Fit (contact_x, slope) by grid search, minimising the cube error against
   XG on half the seeds and scoring the other half, both ways round, plus the
   flat model with the same procedure. Reported against XG and against the
   Sage reference, with paired differences to the analyzer's current table.

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

GRID = [round(0.55 + 0.025 * i, 3) for i in range(19)]
PIVOT = 120.0


def sweep(rows: list[dict], levels: list[str], threads: int) -> None:
    """Fill r["picks"][level][x] = (should_double, should_take)."""
    tl = threading.local()
    saved = A._contact_cube_life, A._contact_cube_slope
    A._contact_cube_life = lambda n, a1=0, a2=0: tl.x
    A._contact_cube_slope = lambda n, a1=0, a2=0: 0.0
    analyzers: dict = {}

    def job(args):
        i, level, x = args
        tl.x = x
        key = (threading.get_ident(), level)
        if key not in analyzers:
            analyzers[key] = BgBotAnalyzer(eval_level=level, cubeful=True, parallel_threads=1)
        r = rows[i]
        a = analyzers[key].cube_action(r["board"], cube_value=r["cube_value"],
                                       cube_owner=r["cube_owner"], jacoby=True, beaver=True)
        return i, level, x, bool(a.should_double), bool(a.should_take)

    try:
        for level in levels:
            jobs = [(i, level, x) for i, r in enumerate(rows) if not r["race"] for x in GRID]
            jobs += [(i, level, -1.0) for i, r in enumerate(rows) if r["race"]]
            with ThreadPoolExecutor(threads) as ex:
                for i, lv, x, sd, st in ex.map(job, jobs, chunksize=64):
                    rows[i]["picks"].setdefault(lv, {})[x] = (sd, st)
            print(f"swept {level}: {len(jobs)} cube analyses", flush=True)
    finally:
        A._contact_cube_life, A._contact_cube_slope = saved


def snap(x: float) -> float:
    return min(GRID, key=lambda g: abs(g - x))


def x_of(r: dict, a: float, b: float) -> float:
    return snap(min(1.0, max(0.55, a + b * (sum(r["pips"]) / 2 - PIVOT))))


def errors(rows, level, a, b, ref):
    out = []
    for r in rows:
        pick = r["picks"][level][-1.0 if r["race"] else x_of(r, a, b)]
        out.append(X.errors(r[ref], r["has_double"], r["has_take"], *pick))
    return out


def summarise(rows, level, model_by_row, ref):
    """PR, blunders and per-position errors with each row's own (a, b)."""
    per, flat = [], []
    for r in rows:
        e = errors([r], level, *model_by_row(r), ref)[0]
        per.append(sum(e))
        flat += e
    return sum(flat) / len(flat) * 500, sum(v > bm.BLUNDER_THRESHOLD for v in flat), per


def paired(a, b, k):
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    return m * k, math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1) / n) * k


def fit(rows, level, slopes):
    total = lambda a, b: sum(sum(e) for e in errors(rows, level, a, b, "xg"))  # noqa: E731
    return min(((total(a, b), a, b) for a, b in
                itertools.product([0.70 + 0.0125 * i for i in range(13)], slopes)))[1:]


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
        pp, op = bgbot_cpp.pip_counts(r["board"])
        rows.append({**{k: r[k] for k in ("board", "cube_value", "cube_owner",
                                          "has_double", "has_take")},
                     "seed": e["seed"], "race": bool(bgbot_cpp.is_race(r["board"])),
                     "pips": (pp, op), "xg": r["xg"], "sage": X.sage_ref(e), "picks": {}})
    n_dec = sum(r["has_double"] + r["has_take"] for r in rows)
    print(f"{len(rows)} positions ({unanalysed} unanalysed by XG), {n_dec} cube decisions")
    sweep(rows, levels, args.threads)

    halves = {h: [r for r in rows if r["seed"] % 2 == h] for h in (0, 1)}
    slope_grid = [0.00025 * i for i in range(-4, 13)]
    for level in levels:
        cur = (A.MONEY_CONTACT_CUBE_LIFE.get(int(level[0]), 0.68),
               A.MONEY_CONTACT_CUBE_LIFE_PIP_SLOPE.get(int(level[0]), 0.0))
        print(f"\n{level} (analyzer table: x = {cur[0]} + {cur[1]} * (avg pips - 120))")
        oof_err = {}
        for name, slopes in (("flat", [0.0]), ("pip", slope_grid)):
            fits = {h: fit(halves[h], level, slopes) for h in (0, 1)}
            all_fit = fit(rows, level, slopes)
            print(f"  {name:4s} fitted on even / odd / all seeds: " + " / ".join(
                f"{a:.4f} + {b:.5f}" for a, b in (fits[0], fits[1], all_fit)))
            # each half scored by the model fitted on the other half
            oof = lambda r, fits=fits: fits[1 - r["seed"] % 2]  # noqa: E731
            for ref in ("xg", "sage"):
                p, b, e = summarise(rows, level, oof, ref)
                oof_err[(name, ref)] = e
                print(f"    out of fold vs {ref:4s}: cube PR {p:.2f} ({b} blunders)")
        for ref in ("xg", "sage"):
            m, se = paired(oof_err[("flat", ref)], oof_err[("pip", ref)], 500 * len(rows) / n_dec)
            print(f"  pip - flat, out of fold, vs {ref:4s}: {m:+.2f} +- {se:.2f}")
        for ref in ("xg", "sage"):
            p, b, _ = summarise(rows, level, lambda r: cur, ref)
            print(f"  analyzer table (in sample) vs {ref:4s}: cube PR {p:.2f} ({b} blunders)")

if __name__ == "__main__":
    main()
