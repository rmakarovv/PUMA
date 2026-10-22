"""Export Sudoku run records to CSV for later analysis (standard library only).

Reads every run directory <exp>/<COND>/seed<S>/ (metrics.jsonl, run_info.txt, config.yaml,
code.diff) and writes to --out:
  evals.csv      one row per evaluation (accuracy of every decoder, val loss, timing)
  train_log.csv  one row per logged training step (loss, lr, train time)
  runs.csv       one row per run (provenance, final metrics, updates/time to accuracy targets)
  summary.md     mean ± std per condition
  raw/<exp>/<COND>/seed<S>/  copies of the small per-run records (no checkpoints)
Usage: python scripts/export_sudoku_results.py [--out results/sudoku] [runs/sudoku_* ...]
"""
import argparse
import csv
import glob
import json
import os
import re
import shutil
import statistics

CONDITION_NAMES = {
    "A": "random masking",
    "E": "confidence progression",
    "F": "Gt-Margin progression",
    "P": "PUMA reference",
    "PN": "PUMA without collapse",
}
TARGETS = (0.5, 0.75, 0.9)
RAW_FILES = ("metrics.jsonl", "run_info.txt", "config.yaml", "code.diff")


def parse_run_info(path):
    info = {}
    if not os.path.exists(path):
        return info
    text = open(path).read()
    for key in ("condition", "seed", "config", "slurm_job", "node", "start", "end", "exit", "git_head", "CUDA_VISIBLE_DEVICES"):
        m = re.search(rf"(?:^|\s){key}=(\S*)", text)
        if m:
            info[key] = m.group(1)
    # every GPU nvidia-smi listed on the node, as index:uuid (allocated ones per CUDA_VISIBLE_DEVICES)
    info["gpus"] = ";".join(f"{i}:{u}" for i, u in re.findall(r"^(\d+), (GPU-[0-9a-f-]+)", text, flags=re.M))
    return info


def first_crossing(evals, decoder, target):
    """First of two consecutive evals with accuracy >= target (proposal rule); None if never."""
    for a, b in zip(evals, evals[1:]):
        if a["val_acc"][decoder] >= target and b["val_acc"][decoder] >= target:
            return a
    return None


def mean_std(xs, fmt):
    if not xs:
        return ""
    return fmt(statistics.mean(xs)) + (" ± " + fmt(statistics.stdev(xs)) if len(xs) > 1 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dirs", nargs="*", help="experiment directories (default: runs/sudoku_*)")
    ap.add_argument("--out", default="results/sudoku")
    ap.add_argument("--decoder", default="top_k_unmasking_1", help="primary decoder for targets and summary")
    args = ap.parse_args()
    exp_dirs = args.exp_dirs or sorted(glob.glob("runs/sudoku_*"))
    os.makedirs(args.out, exist_ok=True)

    eval_rows, train_rows, run_rows = [], [], []
    decoders = set()
    for exp_dir in exp_dirs:
        exp = os.path.basename(os.path.normpath(exp_dir))
        for run_dir in sorted(glob.glob(os.path.join(exp_dir, "*", "seed*"))):
            cond = os.path.basename(os.path.dirname(run_dir))
            seed = int(os.path.basename(run_dir)[len("seed"):])
            metrics = os.path.join(run_dir, "metrics.jsonl")
            if not os.path.exists(metrics):
                continue
            records = [json.loads(line) for line in open(metrics) if line.strip()]
            evals = [r for r in records if "val_loss" in r]
            key = dict(experiment=exp, condition=cond, condition_name=CONDITION_NAMES.get(cond, ""), seed=seed)
            for r in records:
                if "val_loss" in r:
                    decoders.update(r["val_acc"])
                    eval_rows.append({**key, "step": r["step"], "epoch": r["epoch"], "train_time_s": r["train_time_s"],
                                      "eval_time_s": r["eval_time_s"], "val_loss": r["val_loss"],
                                      **{f"acc_{d}": v for d, v in r["val_acc"].items()}})
                else:
                    train_rows.append({**key, "step": r["step"], "epoch": r["epoch"], "loss": r["loss"],
                                       "lr": r["lr"], "train_time_s": r["train_time_s"]})

            info = parse_run_info(os.path.join(run_dir, "run_info.txt"))
            code_diff = os.path.join(run_dir, "code.diff")
            row = {**key, "status": "done" if os.path.exists(os.path.join(run_dir, "DONE")) else "incomplete",
                   "config": info.get("config", ""), "slurm_job": info.get("slurm_job", ""), "node": info.get("node", ""),
                   "cuda_visible_devices": info.get("CUDA_VISIBLE_DEVICES", ""), "gpus": info.get("gpus", ""),
                   "start": info.get("start", ""), "end": info.get("end", ""), "exit": info.get("exit", ""),
                   "git_head": info.get("git_head", ""),
                   "uncommitted_code_diff": int(os.path.exists(code_diff) and os.path.getsize(code_diff) > 0)}
            if evals:
                last = evals[-1]
                row.update(final_step=last["step"], final_val_loss=last["val_loss"],
                           train_time_min=round(last["train_time_s"] / 60, 2),
                           eval_time_min=round(sum(e["eval_time_s"] for e in evals) / 60, 2),
                           **{f"final_acc_{d}": v for d, v in last["val_acc"].items()},
                           best_acc=max(e["val_acc"][args.decoder] for e in evals))
                for t in TARGETS:
                    hit = first_crossing(evals, args.decoder, t)
                    row[f"updates_to_{int(t * 100)}"] = hit["step"] if hit else ""
                    row[f"train_min_to_{int(t * 100)}"] = round(hit["train_time_s"] / 60, 2) if hit else ""
            run_rows.append(row)

            raw_dir = os.path.join(args.out, "raw", exp, cond, f"seed{seed}")
            os.makedirs(raw_dir, exist_ok=True)
            for name in RAW_FILES:
                if os.path.exists(os.path.join(run_dir, name)):
                    shutil.copy2(os.path.join(run_dir, name), raw_dir)

    def write_csv(name, rows):
        fields = []
        for r in rows:
            fields += [k for k in r if k not in fields]
        with open(os.path.join(args.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    write_csv("evals.csv", eval_rows)
    write_csv("train_log.csv", train_rows)
    write_csv("runs.csv", run_rows)

    # summary per condition (all experiments pooled; conditions share seeds and protocol)
    lines = [f"# Sudoku results summary",
             "",
             f"Generated by `scripts/export_sudoku_results.py` from {', '.join(f'`{d}`' for d in exp_dirs)}.",
             f"Accuracy is exact-grid accuracy of the `{args.decoder}` decoder; mean ± std over seeds.",
             "A target counts as reached at the first of two consecutive evals at or above it.",
             "",
             "| Condition | Done runs | Final acc. (%) | Best acc. (%) | Final val loss | Train min | Updates to 50% | Updates to 75% | Updates to 90% |",
             "|---|---|---|---|---|---|---|---|---|"]
    for cond in sorted({r["condition"] for r in run_rows}):
        rs = [r for r in run_rows if r["condition"] == cond and r["status"] == "done"]
        if not rs:
            continue
        cells = [f"{cond} ({CONDITION_NAMES.get(cond, '')})", str(len(rs)),
                 mean_std([100 * r[f"final_acc_{args.decoder}"] for r in rs], lambda x: f"{x:.1f}"),
                 mean_std([100 * r["best_acc"] for r in rs], lambda x: f"{x:.1f}"),
                 mean_std([r["final_val_loss"] for r in rs], lambda x: f"{x:.4f}"),
                 mean_std([r["train_time_min"] for r in rs], lambda x: f"{x:.1f}")]
        for t in TARGETS:
            hits = [r[f"updates_to_{int(t * 100)}"] for r in rs if r[f"updates_to_{int(t * 100)}"] != ""]
            cells.append(mean_std([h / 1000 for h in hits], lambda x: f"{x:.1f}k") if len(hits) == len(rs)
                         else f"reached by {len(hits)}/{len(rs)}")
        lines.append("| " + " | ".join(cells) + " |")
    open(os.path.join(args.out, "summary.md"), "w").write("\n".join(lines) + "\n")
    print(f"wrote {len(eval_rows)} eval rows, {len(train_rows)} train rows, {len(run_rows)} runs to {args.out}")


if __name__ == "__main__":
    main()
