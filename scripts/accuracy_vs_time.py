# SPDX-License-Identifier: MPL-2.0
"""
Checker-play accuracy against thinking time, Open Sage vs GNUbg.

Each engine picks its move for every Benchmark PR decision the way it does
in real use, and the CPU time of that pick is measured:

* Open Sage: ``possible_moves`` + ``MultiPlyStrategy.best_move_index`` (the
  filter chain BgBotAnalyzer uses: 1-ply 5/0.08, plus a 3-ply step at 4-ply,
  PubEval opponent prefilter, bearoff database), one thread per process.
* GNUbg: its own ``hint`` with its default move filters, cubeless,
  ``set threads 1``. Positions are batched into one gnubg process per chunk;
  the chunk's CPU time (from ``wait4``) minus gnubg's start-up cost is spread
  over the chunk's decisions.

Levels use the XG convention (1-ply = raw net) for both engines; GNUbg gets
``plies - 1``. Accuracy is the Benchmark PR: rollout equity lost against the
best rolled-out candidate, x500 (a pick outside the rolled-out set is charged
the worst candidate, as in score_benchmark_pr.py). The reference is Stage 5's
rollout over Stage 5's 2-ply filter set, so it leans towards Open Sage.

Usage:
  python scripts/accuracy_vs_time.py --engine sage --plies 1 2 3
  python scripts/accuracy_vs_time.py --engine gnubg --plies 1 2 3
  python scripts/accuracy_vs_time.py --engine gnubg --plies 4 --sample 4
  python scripts/accuracy_vs_time.py --report
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'build'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'python'))
sys.path.insert(0, SCRIPT_DIR)

import bgbot_cpp  # noqa: E402

OUT_DIR = os.path.join(PROJECT_ROOT, 'logs', 'accuracy_vs_time')
DECISION_STEP = 20  # the rolled-out 1-in-20 sample (CLAUDE.md)


# --------------------------------------------------------------------------
# Data

def load_decisions(sample):
    from score_benchmark_pr import load_benchmark_data
    decisions, _ = load_benchmark_data(DECISION_STEP)
    decisions = [d for d in decisions if len(d['candidates']) >= 2]
    for i, d in enumerate(decisions):
        d['id'] = i
    return decisions[::sample]


def charge(dec, chosen):
    """Rollout equity lost by `chosen` (a board tuple)."""
    eqs = dec['rollout_equities']
    best, worst = max(eqs), min(eqs)
    for cand, eq in zip(dec['candidates'], eqs):
        if tuple(cand) == chosen:
            return best - eq, False
    return best - worst, True


# --------------------------------------------------------------------------
# Open Sage

_SAGE = None


def _sage_init(model, plies):
    global _SAGE
    from bgsage.weights import WeightConfig, bearoff_db_path
    bgbot_cpp.init_escape_tables()
    w = WeightConfig.from_model(model)
    w.validate()
    db = bgbot_cpp.BearoffDB()
    if not db.load(bearoff_db_path()):
        raise RuntimeError('bearoff database missing')
    strat = bgbot_cpp.create_multipy(
        w.strategy_type, w.weight_paths_list, w.hidden_sizes_list,
        n_plies=plies, filter_max_moves=5, filter_threshold=0.08)
    bgbot_cpp.multipy_set_bearoff_db(strat, db)
    _SAGE = (strat, db)


def _sage_chunk(chunk):
    strat, _ = _SAGE
    out = []
    for dec in chunk:
        board, (d1, d2) = dec['board'], dec['dice']
        t0 = time.thread_time()
        moves = bgbot_cpp.possible_moves(board, d1, d2)
        idx = strat.best_move_index(moves, board) if len(moves) > 1 else 0
        cpu = time.thread_time() - t0
        out.append((dec['id'], tuple(moves[idx]), cpu))
    return out


def run_sage(decisions, plies, model, workers, chunk_size):
    chunks = [decisions[i:i + chunk_size]
              for i in range(0, len(decisions), chunk_size)]
    picks = {}
    with ProcessPoolExecutor(workers, initializer=_sage_init,
                             initargs=(model, plies)) as ex:
        for res in ex.map(_sage_chunk, chunks):
            for dec_id, board, cpu in res:
                picks[dec_id] = (board, cpu)
    return picks


# --------------------------------------------------------------------------
# GNUbg

def _gnubg_bin():
    from bgsage.settings import gnubg_path
    return gnubg_path()


def _gnubg_header(plies):
    return ('set threads 1\n'
            'new session\n'
            f'set evaluation chequerplay evaluation plies {plies - 1}\n'
            'set evaluation chequerplay evaluation cubeful off\n')


def _run_gnubg_timed(script):
    """Run gnubg on a command script; return (stdout, child CPU seconds)."""
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
        f.write(script)
        path = f.name
    try:
        with tempfile.TemporaryFile('w+') as out:
            proc = subprocess.Popen([_gnubg_bin(), '-q', '-t', '--no-rc', '-c', path],
                                    stdout=out, stderr=subprocess.DEVNULL)
            _, status, ru = os.wait4(proc.pid, 0)
            proc.returncode = os.waitstatus_to_exitcode(status)
            out.seek(0)
            text = out.read()
    finally:
        os.remove(path)
    return text, ru.ru_utime + ru.ru_stime


_TOP_MOVE = re.compile(r'^\s+1\.\s+Cube\w+\s+\S+\s+(.+?)\s+Eq\.:', re.M)


def _apply_gnubg_move(board, text):
    """Apply GNUbg move notation (player on roll, points 1-24, bar=25)."""
    b = list(board)

    def point(tok):
        tok = tok.rstrip('*')
        if tok == 'bar':
            return 25
        if tok == 'off':
            return 0
        return int(tok)

    for token in text.split():
        count = 1
        m = re.fullmatch(r'(.+)\((\d)\)', token)
        if m:
            token, count = m.group(1), int(m.group(2))
        steps = [point(p) for p in token.split('/')]
        for _ in range(count):
            for src, dst in zip(steps, steps[1:]):
                b[src] -= 1
                if dst == 0:
                    continue
                if b[dst] == -1:          # hit
                    b[dst] = 0
                    b[0] += 1
                b[dst] += 1
    return tuple(b)


def _gnubg_chunk(args):
    chunk, plies = args
    script = _gnubg_header(plies)
    for dec in chunk:
        b = dec['board']
        script += ('set board simple ' + ' '.join(str(v) for v in
                   [b[25]] + b[1:25] + [b[0]]) + '\n')
        script += f"set dice {dec['dice'][0]} {dec['dice'][1]}\nhint\n"
    text, cpu = _run_gnubg_timed(script)
    tops = _TOP_MOVE.findall(text)
    if len(tops) != len(chunk):
        raise RuntimeError(f'gnubg returned {len(tops)} hints for {len(chunk)} positions')
    out = []
    for dec, move in zip(chunk, tops):
        legal = {tuple(m) for m in bgbot_cpp.possible_moves(dec['board'], *dec['dice'])}
        board = _apply_gnubg_move(dec['board'], move)
        if board not in legal:
            raise RuntimeError(f'could not apply gnubg move {move!r} to {dec["board"]}')
        out.append((dec['id'], board))
    return out, cpu


def gnubg_startup_cpu(plies, n=5):
    return min(_run_gnubg_timed(_gnubg_header(plies))[1] for _ in range(n))


def run_gnubg(decisions, plies, workers, chunk_size):
    bgbot_cpp.init_escape_tables()
    startup = gnubg_startup_cpu(plies)
    chunks = [decisions[i:i + chunk_size]
              for i in range(0, len(decisions), chunk_size)]
    picks = {}
    with ThreadPoolExecutor(workers) as ex:
        for res, cpu in ex.map(_gnubg_chunk, [(c, plies) for c in chunks]):
            per = max(cpu - startup, 0.0) / len(res)
            for dec_id, board in res:
                picks[dec_id] = (board, per)
    return picks, startup


# --------------------------------------------------------------------------
# Scoring and report

def summarise(decisions, picks):
    errs, cpus, outside = [], [], 0
    per_dec, outside_ids = {}, []
    for dec in decisions:
        board, cpu = picks[dec['id']]
        err, out = charge(dec, board)
        outside += out
        if out:
            outside_ids.append(dec['id'])
        errs.append(err)
        cpus.append(cpu)
        per_dec[dec['id']] = err
    n = len(errs)
    mean = sum(errs) / n
    sd = math.sqrt(sum((e - mean) ** 2 for e in errs) / (n - 1))
    return {
        'n': n,
        'pr': mean * 500,
        'pr_se': sd / math.sqrt(n) * 500,
        'cpu_ms_mean': sum(cpus) / n * 1000,
        'outside': outside,
        'outside_ids': outside_ids,
        'errors': per_dec,
    }


def report():
    rows = []
    for name in sorted(os.listdir(OUT_DIR)):
        if name.endswith('.json'):
            with open(os.path.join(OUT_DIR, name)) as f:
                rows.append(json.load(f))
    rows.sort(key=lambda r: (r['engine'], r['plies'], r['sample']))
    print(f"{'engine':<8}{'level':>6}{'n':>7}{'PR':>8}{'±SE':>7}"
          f"{'CPU ms/dec':>12}{'outside':>9}")
    for r in rows:
        print(f"{r['engine']:<8}{str(r['plies']) + 'P':>6}{r['n']:>7}"
              f"{r['pr']:>8.2f}{r['pr_se']:>7.2f}{r['cpu_ms_mean']:>12.2f}"
              f"{r['outside']:>9}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--engine', choices=['sage', 'gnubg'])
    ap.add_argument('--plies', type=int, nargs='+', default=[1, 2])
    ap.add_argument('--model', default=None, help='Open Sage model (default: production)')
    ap.add_argument('--sample', type=int, default=1,
                    help='score every Nth decision of the rolled-out sample')
    ap.add_argument('--workers', type=int, default=10,
                    help='parallel single-threaded workers (default 10 = P-cores)')
    ap.add_argument('--chunk', type=int, default=0)
    ap.add_argument('--report', action='store_true')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.report:
        report()
        return
    if args.engine is None:
        ap.error('--engine is required')

    from bgsage.weights import PRODUCTION_MODEL
    model = args.model or PRODUCTION_MODEL
    decisions = load_decisions(args.sample)
    print(f'{len(decisions)} decisions (sample 1/{args.sample} of the 1-in-{DECISION_STEP} set)')

    for plies in args.plies:
        t0 = time.perf_counter()
        extra = {}
        if args.engine == 'sage':
            picks = run_sage(decisions, plies, model, args.workers, args.chunk or 25)
            label = model
        else:
            picks, startup = run_gnubg(decisions, plies, args.workers, args.chunk or 40)
            extra['gnubg_startup_cpu_s'] = startup
            label = 'gnubg'
        s = summarise(decisions, picks)
        s.update(engine=args.engine, model=label, plies=plies, sample=args.sample,
                 wall_s=time.perf_counter() - t0, **extra)
        print(f"{args.engine} {plies}-ply: PR {s['pr']:.2f} ± {s['pr_se']:.2f}  "
              f"CPU {s['cpu_ms_mean']:.2f} ms/decision  outside {s['outside']}  "
              f"[{s['wall_s']:.0f}s]", flush=True)
        name = f"{args.engine}_{plies}ply_s{args.sample}.json"
        with open(os.path.join(OUT_DIR, name), 'w') as f:
            json.dump(s, f)


if __name__ == '__main__':
    main()
