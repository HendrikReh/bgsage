#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""Score the batch cube path's contact cube life against XG.

``batch_evaluate`` / ``batch_cube_action`` (and ``batch_post_move_evaluate``)
convert cubeless probs to cube decisions with one Janowski step at the root,
at every depth: N-ply probs from 2-ply on, no cube tree. This script plays the
money benchmark's cube decisions through that path at a fixed depth with
several contact cube lives and scores them against XG's verdicts
(``data/money_cube_xg/xg_reference.jsonl``, written by
``export_money_cube_xg.py score``) and against the Sage reference, by seed half,
with paired differences against the engine's default (0.68).

Candidates: the engine default, the analyzer's 1-ply and 2-ply money values
(``MONEY_CONTACT_CUBE_LIFE`` / ``_PIP_SLOPE``), and with ``--grid`` a
(value, slope) grid plus an out-of-fold fit (fitted on one seed half against
XG, scored on the other). Scoring is ``export_money_cube_xg.errors``.

Usage::

    python scripts/score_batch_cube_life_xg.py --levels 2 3 4
    python scripts/score_batch_cube_life_xg.py --levels 2 --grid
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_SCRIPT_DIR, _PROJECT_ROOT / "build", _PROJECT_ROOT / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import bgbot_cpp  # noqa: E402
import export_money_cube_xg as mx  # noqa: E402
from bgsage import BatchModels  # noqa: E402
from bgsage.analyzer import (  # noqa: E402
    MONEY_CONTACT_CUBE_LIFE, MONEY_CONTACT_CUBE_LIFE_PIP_SLOPE, resolve_owner,
)

BLUNDER = 0.08
GRID = [(v, s) for v in (0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82)
        for s in (0.0, 0.00125, 0.00175)]


class Scorer:
    def __init__(self, xg_dir: Path, threads: int):
        self.rows = [json.loads(line) for line in open(xg_dir / "xg_reference.jsonl")
                     if line.strip()]
        by_key = {e["key"]: e for e in mx.load_cube_entries()}
        self.half = [by_key[r["key"]]["seed"] % 2 for r in self.rows]
        self.refs = {"xg": [r["xg"] for r in self.rows],
                     "sage": [mx.sage_ref(by_key[r["key"]]) for r in self.rows]}
        self.n_dec = sum(r["has_double"] + r["has_take"] for r in self.rows)
        self.cpp = [(r["board"], r["cube_value"], resolve_owner(r["cube_owner"]))
                    for r in self.rows]
        self.models = BatchModels()
        self.threads = threads

    def picks(self, n_plies: int, x: float, slope: float):
        strategy = self.models.base() if n_plies == 1 else self.models.multipy(n_plies)
        res = bgbot_cpp.batch_evaluate_positions(
            self.cpp, strategy, self.threads, jacoby=True, beaver=True,
            contact_x=x, contact_x_pip_slope=slope)
        return [(bool(r["should_double"]), bool(r["should_take"])) for r in res]

    def errors(self, picks, ref, half=None):
        out = []
        for i, (r, rf, (sd, st)) in enumerate(zip(self.rows, self.refs[ref], picks)):
            if half is None or self.half[i] == half:
                out += mx.errors(rf, r["has_double"], r["has_take"], sd, st)
        return out

    def per_position(self, picks, ref):
        return [sum(mx.errors(rf, r["has_double"], r["has_take"], sd, st))
                for r, rf, (sd, st) in zip(self.rows, self.refs[ref], picks)]

    def paired(self, base, picks, ref):
        d = [y - x for x, y in zip(self.per_position(base, ref), self.per_position(picks, ref))]
        m = sum(d) / len(d)
        se = math.sqrt(sum((v - m) ** 2 for v in d) / (len(d) - 1) / len(d))
        scale = len(self.rows) / self.n_dec * mx.PR_MULTIPLIER   # per-decision PR units
        return m * scale, se * scale

    def line(self, name, picks, base=None):
        cells = []
        for ref in ("xg", "sage"):
            e = self.errors(picks, ref)
            cell = (f"{mx.pr(e):5.2f} ({sum(v > BLUNDER for v in e):3d}) "
                    f"even {mx.pr(self.errors(picks, ref, 0)):5.2f} "
                    f"odd {mx.pr(self.errors(picks, ref, 1)):5.2f}")
            if base is not None:
                cell += "  d {:+5.2f} +- {:4.2f}".format(*self.paired(base, picks, ref))
            cells.append(cell)
        print(f"{name:34s} | vs XG {cells[0]} | vs Sage {cells[1]}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--levels", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--xg-dir", type=Path, default=_PROJECT_ROOT / "data" / "money_cube_xg")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--grid", action="store_true",
                    help="also score a (value, slope) grid and its out-of-fold fit")
    args = ap.parse_args()

    sc = Scorer(args.xg_dir, args.threads)
    print(f"{len(sc.rows)} positions, {sc.n_dec} cube decisions. Cube PR (blunders > "
          f"{BLUNDER}), by seed half; d = paired difference against the engine default.")
    for n in args.levels:
        print(f"\n{n}-ply batch path")
        base = sc.picks(n, -1.0, 0.0)
        sc.line("engine default (0.68)", base)
        for ply in (1, 2):
            x, s = MONEY_CONTACT_CUBE_LIFE[ply], MONEY_CONTACT_CUBE_LIFE_PIP_SLOPE[ply]
            sc.line(f"analyzer {ply}-ply value ({x}, {s})", sc.picks(n, x, s), base)
        if not args.grid:
            continue
        grid = {(x, s): sc.picks(n, x, s) for x, s in GRID}
        for (x, s), p in grid.items():
            sc.line(f"  {x:.4f} + {s:.5f}/pip", p, base)
        oof = [None] * len(sc.rows)
        for h in (0, 1):
            x, s = min(grid, key=lambda k: mx.pr(sc.errors(grid[k], "xg", h)))
            print(f"  fitted on the {'even' if h == 0 else 'odd'} seeds: {x:.4f} + {s:.5f}/pip")
            for i in range(len(sc.rows)):
                if sc.half[i] != h:
                    oof[i] = grid[(x, s)][i]
        sc.line("out-of-fold fit", oof, base)


if __name__ == "__main__":
    main()
