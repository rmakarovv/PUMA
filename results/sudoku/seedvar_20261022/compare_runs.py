"""Compare the fp32-eval B/D retrain (runs/sudoku_<C>_20261022) with the original runs (runs/sudoku_<C>_20261019).

Per seed: max |diff| of train loss and val loss over the steps both logged, and new logged top_k_unmasking_1 vs the
fp32 rescore of the old checkpoints at 10k..final. Then mean, sd and stderr (sd / sqrt(n)) of final top_k_1 per
condition, over the old 5 seeds (rescore) and over all finished new seeds.
usage: python compare_runs.py <runs_dir> <rescore_dir>
"""
import json, math, os, statistics, sys

runs, rescore = sys.argv[1], sys.argv[2]
DEC = "top_k_unmasking_1"


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


def summary(xs):
    sd = statistics.stdev(xs) if len(xs) > 1 else float("nan")
    return f"n={len(xs)} mean {statistics.mean(xs):.1f} sd {sd:.1f} stderr {sd / math.sqrt(len(xs)):.1f}"


for c in ("B", "D"):
    old_final, new_final = [], []
    for s in range(1000, 10001, 1000):
        new = load(f"{runs}/sudoku_{c}_20261022/seed{s}/metrics.jsonl")
        done = os.path.exists(f"{runs}/sudoku_{c}_20261022/seed{s}/DONE")
        old = load(f"{runs}/sudoku_{c}_20261019/seed{s}/metrics.jsonl")
        rs = {r["step"]: r["fp32"][DEC] for r in load(f"{rescore}/{c}_seed{s}.jsonl")}
        if rs:
            old_final.append(100 * rs[max(rs)])
        if not new:
            continue
        ne = {r["step"]: r for r in new if "val_acc" in r}
        line = f"{c} seed{s} {'DONE' if done else 'running'} last_step={new[-1]['step']} evals={len(ne)}"
        if old:
            ot = {r["step"]: r["loss"] for r in old if "loss" in r}
            nt = {r["step"]: r["loss"] for r in new if "loss" in r}
            ov = {r["step"]: r["val_loss"] for r in old if "val_loss" in r}
            common_t, common_v = sorted(set(ot) & set(nt)), sorted(set(ov) & set(ne))
            line += f" | train-loss max|d| {max(abs(ot[k] - nt[k]) for k in common_t):.2e} over {len(common_t)}"
            line += f" | val-loss max|d| {max(abs(ov[k] - ne[k]['val_loss']) for k in common_v):.2e} over {len(common_v)}"
        if rs:
            pairs = [(k, 100 * ne[k]["val_acc"][DEC], 100 * rs[k]) for k in sorted(rs) if k in ne]
            line += " | new vs rescore: " + " ".join(f"{k // 1000}k {a:.1f}/{b:.1f}" for k, a, b in pairs)
        print(line)
        if done:
            new_final.append(100 * ne[max(ne)]["val_acc"][DEC])
    print(f"{c} old fp32 rescore final: {summary(old_final)}")
    if len(new_final) > 1:
        print(f"{c} new logged final:       {summary(new_final)}")
