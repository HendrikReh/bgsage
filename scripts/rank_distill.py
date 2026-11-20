#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""Go/no-go test: does a within-decision (ranking) loss improve checker play?

Two retraining runs (pure race, containment) lowered the nets' holdout error
against better targets without improving their picks: MSE on absolute
probabilities mostly fits the part of the error that every candidate of a
decision shares, and that part cancels in the decision. This script tests the
alternative cheaply, with NeonGammon's own 3-ply as the teacher instead of
rollouts:

* ``mine``  plays stage11 1-ply self-play games (with some exploration), keeps
  the decisions whose 1-ply candidates are close (top <= 6 within 0.12 of the
  best), and labels every kept candidate with its 3-ply cubeless post-move
  evaluation. Decisions whose pre-move board is in the Benchmark PR sample
  are dropped, so the XG check below stays held out.
* ``train`` fine-tunes the contact pair nets on those decisions with MLX,
  under one of two losses:
    - ``abs``:  MSE of the five probabilities against the 3-ply ones (the
      control: plain distillation, the loss every earlier run used);
    - ``rank``: MSE of the candidates' equities CENTRED on their decision's
      mean, against the centred 3-ply equities, plus ``--anchor`` x the abs
      loss so the absolute level does not drift (2-ply compares leaves of
      different decisions, so it needs the level).
  1-ply checker play routes every candidate to the net of the pre-move board
  (``evaluate_probs(board, pre_move_board)``), so all candidates of a decision
  go through one net and each net trains on its own decisions.
* ``score`` plays the Benchmark PR sample's XG-analysed decisions at 1-ply and
  2-ply with the production nets and with any trained variants, and prints PR
  against XG 4-ply and against the Stage 5 rollout, with paired differences.

Outputs land in ``logs/rank_distill/`` (mined data) and
``models/rank_diag/<variant>/`` (weights; untracked).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_SCRIPT_DIR, _PROJECT_ROOT / "build", _PROJECT_ROOT / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import bgbot_cpp  # noqa: E402
from neongammon import STARTING_BOARD, possible_moves, flip_board, check_game_over  # noqa: E402
from neongammon.sl_train import read_weights, write_weights  # noqa: E402
from neongammon.weights import WeightConfig, bearoff_db_path  # noqa: E402

DATA_DIR = _PROJECT_ROOT / "logs" / "rank_distill"
MODEL_DIR = _PROJECT_ROOT / "models" / "rank_diag"
XG_DIRS = [_PROJECT_ROOT / "data" / "benchmark_pr_xg" / d for d in ("contact_xg4ply", "race_xg4ply")]
PR = 500.0
K_MAX = 6
WINDOW = 0.12
N_CONTACT_SLOTS = 17          # 0 purerace, 1-16 the contact pair nets


def equity(p: np.ndarray) -> np.ndarray:
    """Cubeless money equity of [..., 5] probabilities."""
    return 2 * p[..., 0] - 1 + p[..., 1] - p[..., 3] + p[..., 2] - p[..., 4]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Nets:
    """The production model's nets in NumPy, routed like the engine's 1-ply."""

    def __init__(self, w: WeightConfig, overrides: dict[int, str] | None = None):
        self.w = w
        self.paths = list(w.weight_paths_list)
        for slot, path in (overrides or {}).items():
            self.paths[slot] = path
        self.hidden = list(w.hidden_sizes_list)
        self.strategy = bgbot_cpp.create_strategy(w.strategy_type, self.paths, self.hidden)
        self._cache: dict[str, tuple] = {}

    def net(self, slot: int):
        p = self.paths[slot]
        if p not in self._cache:
            self._cache[p] = read_weights(p)
        return self._cache[p]

    def route(self, pre_board) -> int:
        return int(self.strategy.select_nn_idx(list(pre_board)))

    def probs(self, slot: int, boards: np.ndarray) -> np.ndarray:
        n_in = 196 if slot == 0 else 244
        x = bgbot_cpp.encode_boards_batch(boards.astype(np.int32), n_in)
        w_h, b_h, w_o, b_o = self.net(slot)
        return sigmoid(sigmoid(x @ w_h.T + b_h) @ w_o.T + b_o)


# ---------------------------------------------------------------------------
# mine
# ---------------------------------------------------------------------------

def benchmark_boards() -> set[tuple]:
    out = set()
    for d in XG_DIRS:
        f = d / "xg_reference.jsonl"
        if f.exists():
            for line in open(f):
                out.add(tuple(json.loads(line)["board"]))
    return out


def roll(rng):
    return rng.randint(1, 6), rng.randint(1, 6)


def cmd_mine(args) -> None:
    from neongammon.batch import BatchModels, batch_post_move_evaluate

    rng = random.Random(args.seed)
    w = WeightConfig.default()
    nets = Nets(w)
    db = bgbot_cpp.BearoffDB()
    if not db.load(bearoff_db_path() or ""):
        raise SystemExit("bearoff database not found")
    excluded = benchmark_boards()
    print(f"{len(excluded)} Benchmark PR boards excluded")

    seen: set[tuple] = set()
    decisions = []          # dict(pre, slot, cands[k,26], game)
    t0 = time.time()
    game = 0
    while len(decisions) < args.decisions:
        board = list(STARTING_BOARD)
        d1, d2 = roll(rng)
        while d1 == d2:
            d1, d2 = roll(rng)
        for _ in range(400):
            cands = possible_moves(board, d1, d2)
            if not cands:
                board = flip_board(board)
                d1, d2 = roll(rng)
                continue
            if len(cands) == 1:
                nxt = cands[0]
            else:
                slot = nets.route(board)
                arr = np.array(cands, dtype=np.int32)
                eq = equity(nets.probs(slot, arr))
                order = np.argsort(-eq)
                # terminal candidates: a move that wins the game is always best
                for i, c in enumerate(cands):
                    r = check_game_over(c)
                    if r:
                        eq[i] = 10 + abs(r)
                order = np.argsort(-eq)
                keep = [int(i) for i in order[:K_MAX] if eq[order[0]] - eq[i] <= WINDOW]
                key = tuple(board)
                if (len(keep) >= 2 and 1 <= slot < N_CONTACT_SLOTS and key not in seen
                        and key not in excluded and not db.is_bearoff(board)
                        and eq[order[0]] < 5):
                    seen.add(key)
                    decisions.append({"pre": board, "slot": slot, "game": game,
                                      "cands": arr[keep]})
                if rng.random() < args.explore:
                    nxt = cands[int(order[rng.randrange(min(3, len(order)))])]
                else:
                    nxt = cands[int(order[0])]
            if check_game_over(nxt):
                break
            board = flip_board(nxt)
            d1, d2 = roll(rng)
        game += 1
        if game % 200 == 0:
            print(f"  {game} games, {len(decisions)} decisions, {time.time() - t0:.0f}s", flush=True)
    decisions = decisions[: args.decisions]
    print(f"mined {len(decisions)} decisions from {game} games in {time.time() - t0:.0f}s")

    # label every candidate at 3-ply (cubeless post-move)
    flat = [c for d in decisions for c in d["cands"]]
    print(f"labelling {len(flat)} candidates at {args.teacher} ...", flush=True)
    models = BatchModels(w)
    t1 = time.time()
    probs = np.zeros((len(flat), 5), dtype=np.float32)
    chunk = 20000
    for s in range(0, len(flat), chunk):
        res = batch_post_move_evaluate(
            [{"board": [int(v) for v in b], "cube_owner": "centered"} for b in flat[s:s + chunk]],
            eval_level=args.teacher, models=models, n_threads=0)
        for i, r in enumerate(res):
            p = r.probs
            probs[s + i] = (p.win, p.gammon_win, p.backgammon_win, p.gammon_loss, p.backgammon_loss)
        print(f"  {min(s + chunk, len(flat))}/{len(flat)}  {time.time() - t1:.0f}s", flush=True)

    n_c = np.array([len(d["cands"]) for d in decisions], dtype=np.int32)
    out = DATA_DIR / f"{args.name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        pre=np.array([d["pre"] for d in decisions], dtype=np.int8),
        slot=np.array([d["slot"] for d in decisions], dtype=np.int8),
        game=np.array([d["game"] for d in decisions], dtype=np.int32),
        n_c=n_c,
        cands=np.concatenate([d["cands"] for d in decisions]).astype(np.int8),
        probs=probs,
        teacher=np.array(args.teacher))
    print(f"-> {out}")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def load_mined(names: list[str]):
    parts = [np.load(DATA_DIR / f"{n}.npz") for n in names]
    out = {}
    game_off = 0
    for p in parts:
        for k in ("pre", "slot", "n_c", "cands", "probs"):
            out.setdefault(k, []).append(p[k])
        out.setdefault("game", []).append(p["game"] + game_off)
        game_off += int(p["game"].max()) + 1
    return {k: np.concatenate(v) for k, v in out.items()}


def group_by_file(data, w: WeightConfig):
    """Per weight file: padded decision tensors X [D,K,244], T [D,K,5], M [D,K]."""
    paths = list(w.weight_paths_list)
    offs = np.concatenate([[0], np.cumsum(data["n_c"])])
    by_file: dict[str, list[int]] = {}
    for i, s in enumerate(data["slot"]):
        by_file.setdefault(paths[int(s)], []).append(i)
    groups = {}
    for path, idx in by_file.items():
        D = len(idx)
        X = np.zeros((D, K_MAX, 244), dtype=np.float32)
        T = np.zeros((D, K_MAX, 5), dtype=np.float32)
        M = np.zeros((D, K_MAX), dtype=np.float32)
        G = np.zeros(D, dtype=np.int64)
        for j, i in enumerate(idx):
            a, b = offs[i], offs[i + 1]
            X[j, : b - a] = bgbot_cpp.encode_boards_batch(data["cands"][a:b].astype(np.int32), 244)
            T[j, : b - a] = data["probs"][a:b]
            M[j, : b - a] = 1
            G[j] = data["game"][i]
        groups[path] = (X, T, M, G)
    return groups


def cmd_train(args) -> None:
    import mlx.core as mx
    import mlx.optimizers as optim

    w = WeightConfig.default()
    data = load_mined(args.data)
    groups = group_by_file(data, w)
    out_dir = MODEL_DIR / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    report = {}

    def forward(params, x):
        h = mx.sigmoid(x @ params["w_h"].T + params["b_h"])
        return mx.sigmoid(h @ params["w_o"].T + params["b_o"])

    def eq_mx(p):
        return 2 * p[..., 0] - 1 + p[..., 1] - p[..., 3] + p[..., 2] - p[..., 4]

    def loss_fn(params, x, t, m):
        p = forward(params, x)                         # [B,K,5]
        n = m.sum(axis=1, keepdims=True)               # [B,1]
        abs_l = (((p - t) ** 2).sum(axis=-1) * m).sum() / m.sum()
        if args.loss == "abs":
            return abs_l
        e, te = eq_mx(p), eq_mx(t)
        ec = e - (e * m).sum(axis=1, keepdims=True) / n
        tc = te - (te * m).sum(axis=1, keepdims=True) / n
        rank_l = (((ec - tc) ** 2) * m).sum() / m.sum()
        return rank_l + args.anchor * abs_l

    grad_fn = mx.value_and_grad(loss_fn)

    def regret(params, X, T, M):
        """Mean 3-ply equity lost by the net's pick, and the teacher-level loss."""
        p = np.array(forward(params, mx.array(X)))
        e = np.where(M > 0, equity(p), -1e9)
        te = np.where(M > 0, equity(T), -1e9)
        pick = e.argmax(axis=1)
        return float((te.max(axis=1) - te[np.arange(len(te)), pick]).mean())

    for path, (X, T, M, G) in sorted(groups.items(), key=lambda kv: -len(kv[1][0])):
        name = Path(path).name.replace(".weights.best", "")
        games = np.unique(G)
        val_games = set(rng.choice(games, size=max(1, len(games) // 10), replace=False).tolist())
        vmask = np.array([g in val_games for g in G])
        Xt, Tt, Mt = X[~vmask], T[~vmask], M[~vmask]
        Xv, Tv, Mv = X[vmask], T[vmask], M[vmask]
        w_h, b_h, w_o, b_o = read_weights(path)
        params = {k: mx.array(v) for k, v in
                  (("w_h", w_h), ("b_h", b_h), ("w_o", w_o), ("b_o", b_o))}
        opt = optim.Adam(learning_rate=args.lr)
        base = regret(params, Xv, Tv, Mv)
        best = (base, 0, {k: np.array(v) for k, v in params.items()})
        n = len(Xt)
        if n < args.min_decisions:
            print(f"{name:24} {n:6} train decisions: too few, kept unchanged")
            report[name] = {"n_train": n, "skipped": True}
            continue
        for ep in range(1, args.epochs + 1):
            perm = rng.permutation(n)
            for s in range(0, n, args.batch):
                b = perm[s:s + args.batch]
                _, g = grad_fn(params, mx.array(Xt[b]), mx.array(Tt[b]), mx.array(Mt[b]))
                opt.update(params, g)
                mx.eval(params, opt.state)
            r = regret(params, Xv, Tv, Mv)
            if r < best[0]:
                best = (r, ep, {k: np.array(v) for k, v in params.items()})
        print(f"{name:24} train {n:6}  val {len(Xv):5}  val regret x500: "
              f"{base * PR:6.2f} -> {best[0] * PR:6.2f} (epoch {best[1]})", flush=True)
        bw = best[2]
        write_weights(str(out_dir / Path(path).name), bw["w_h"], bw["b_h"], bw["w_o"], bw["b_o"])
        report[name] = {"n_train": n, "n_val": int(len(Xv)), "val_regret_before": base,
                        "val_regret_after": best[0], "best_epoch": best[1]}
    (out_dir / "report.json").write_text(json.dumps(
        {"args": vars(args) | {"data": args.data}, "nets": report}, indent=1))
    print(f"-> {out_dir}")


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def variant_paths(w: WeightConfig, variant: str | None) -> list[str]:
    paths = list(w.weight_paths_list)
    if variant:
        d = MODEL_DIR / variant
        paths = [str(d / Path(p).name) if (d / Path(p).name).exists() else p for p in paths]
    return paths


def cmd_score(args) -> None:
    import export_benchmark_pr_xg as ex

    rows = []
    for d in XG_DIRS:
        f = d / "xg_reference.jsonl"
        rows += [json.loads(line) for line in open(f)]
    rows = [r for r in rows if ex.xg_decision(r) is not None]
    refs = {"XG": [ex.xg_decision(r) for r in rows],
            "S5 RO": [{k: r[k] for k in ex.REF_KEYS} for r in rows]}
    contact = np.array([r["plan"] != "purerace" for r in rows])
    print(f"{len(rows)} decisions ({contact.sum()} contact)")

    bgbot_cpp.init_escape_tables()
    db = bgbot_cpp.BearoffDB()
    if not db.load(bearoff_db_path() or ""):
        raise SystemExit("bearoff database not found")
    w = WeightConfig.default()
    hidden = list(w.hidden_sizes_list)
    errs: dict[str, dict[str, np.ndarray]] = {}
    for variant in [None] + args.variant:
        paths = variant_paths(w, variant)
        base = bgbot_cpp.BearoffStrategy(bgbot_cpp.create_strategy(w.strategy_type, paths, hidden), db)
        strats = [(1, base, None)]
        for plies in args.plies:
            if plies > 1:
                m = bgbot_cpp.create_multipy(w.strategy_type, paths, hidden, n_plies=plies)
                bgbot_cpp.multipy_set_bearoff_db(m, db)
                strats.append((plies, m, base))
        for plies, strat, filt in strats:
            if plies not in args.plies:
                continue
            label = f"{variant or 'stage11'} {plies}P"
            errs[label] = {}
            t0 = time.time()
            for ref, decs in refs.items():
                def one(dec, strat=strat, filt=filt):
                    res = bgbot_cpp.score_benchmark_pr_strategy([dec], strat, filt, n_threads=1)
                    return res["total_error"] if res["total_decisions"] else np.nan
                with ThreadPoolExecutor(max_workers=args.threads) as pool:
                    errs[label][ref] = np.array(list(pool.map(one, decs, chunksize=16)))
            print(f"  {label} done ({time.time() - t0:.0f}s)", flush=True)

    for ref in refs:
        print(f"\nPR against {ref}  (contact decisions / all; paired diff vs stage11 at the same ply)")
        for label, per in errs.items():
            e = per[ref]
            plies = label.split()[-1]
            b = errs[f"stage11 {plies}"][ref]
            line = f"  {label:28} {np.nanmean(e[contact]) * PR:6.3f} / {np.nanmean(e) * PR:6.3f}"
            if not label.startswith("stage11 "):
                d = (e - b)[contact]
                d = d[~np.isnan(d)]
                line += (f"   diff {d.mean() * PR:+.3f} +- {d.std(ddof=1) / math.sqrt(len(d)) * PR:.3f}"
                         f"  changed {int((d != 0).sum())} (better {int((d < 0).sum())})")
            print(line)
    out = DATA_DIR / "score.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({k: {r: v.tolist() for r, v in per.items()} for k, per in errs.items()}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mine")
    m.add_argument("--decisions", type=int, default=50000)
    m.add_argument("--explore", type=float, default=0.15)
    m.add_argument("--teacher", default="3ply")
    m.add_argument("--seed", type=int, default=1)
    m.add_argument("--name", default="mined")
    t = sub.add_parser("train")
    t.add_argument("variant")
    t.add_argument("--data", nargs="+", default=["mined"])
    t.add_argument("--loss", choices=("abs", "rank"), required=True)
    t.add_argument("--anchor", type=float, default=0.1)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--min-decisions", type=int, default=300)
    t.add_argument("--seed", type=int, default=1)
    s = sub.add_parser("score")
    s.add_argument("--variant", action="append", default=[])
    s.add_argument("--plies", type=int, nargs="+", default=[1, 2])
    s.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    bgbot_cpp.init_escape_tables()
    {"mine": cmd_mine, "train": cmd_train, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
