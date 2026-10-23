"""Export Sudoku run records to CSV for later analysis (standard library only).

Reads every run directory <exp>/<COND>/seed<S>/ (metrics.jsonl, run_info.txt, config.yaml,
code.diff) and writes to --out:
  evals.csv      one row per evaluation (accuracy of every decoder, val loss, timing)
  train_log.csv  one row per logged training step (loss, lr, train time)
  runs.csv       one row per run (provenance, final metrics, updates/time to accuracy targets)
  reeval.csv     re-scores of saved EMA checkpoints by scripts/reeval_sudoku.py (runs/reeval_*)
  summary.md     mean ± std per condition (training-time eval, plus the re-score table if any)
  raw/<exp>/<COND>/seed<S>/  copies of the small per-run records (no checkpoints)
  raw/<reeval>/  re-evaluation outputs and job records
Runs trained elsewhere can be added as W&B-style packages (reports/sudoku_*/runs.json: per-run config,
history and summary, accuracies in [0, 1]); their fp32 checkpoint scores go to reeval.csv.
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
    "B": "schedule only",
    "C": "reuse only",
    "D": "random progression",
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


def load_package_runs(path):
    """Convert a W&B-style package (runs.json) into (experiment, condition, seed, records, info, fp32 rows).

    records follow metrics.jsonl (train rows; eval rows with the bf16 eval as val_acc); fp32 rows hold the
    checkpoint re-scores (eval_fp32/*, eval_bf16_fp32_softmax/*) in the reeval.csv layout.
    """
    package = json.load(open(path))
    assert package.get("accuracy_scale", "0-1") == "0-1", path
    for run in package["runs"]:
        cfg = run["config"]
        exp, cond, seed = cfg["experiment"], cfg["condition"], int(cfg["seed"])
        records, rescored = [], []
        history = sorted(run["history"], key=lambda h: h["optimizer_step"])
        final_step = max(h["optimizer_step"] for h in history)
        for h in history:
            step = h["optimizer_step"]
            if "train/loss" in h:
                records.append({"step": step, "epoch": h["epoch"], "loss": h["train/loss"], "lr": h["train/lr"],
                                "train_time_s": h["train/time_s"]})
            if "eval/val_loss" in h:
                acc = {k.split("/", 1)[1][len("acc_"):]: v for k, v in h.items() if k.startswith("eval_bf16/acc_")}
                records.append({"step": step, "epoch": h["epoch"], "val_loss": h["eval/val_loss"], "val_acc": acc,
                                "train_time_s": h["train/time_s"], "eval_time_s": h["eval/time_s"]})
            for prefix, precision in (("eval_fp32/", "fp32"), ("eval_bf16_fp32_softmax/", "bf16_fp32_softmax")):
                for k, v in h.items():
                    if k.startswith(prefix + "acc_"):
                        decoder = k[len(prefix + "acc_"):]
                        rescored.append({"step": step, "final": int(step == final_step), "precision": precision,
                                         "decoder": decoder, "acc": v,
                                         "logged_bf16_acc": h.get(f"eval_bf16/acc_{decoder}", ""),
                                         "eval_time_s": h.get(prefix + "time_s", ""), "node": "", "gpu": ""})
        info = {k: str(cfg.get(k, "")) for k in ("slurm_job", "node", "start", "end", "exit", "git_head")}
        info["uncommitted_code_diff"] = int(bool(cfg.get("source_code_diff_sha256")))
        yield exp, cond, seed, records, info, rescored


def mean_std(xs, fmt):
    if not xs:
        return ""
    return fmt(statistics.mean(xs)) + (" ± " + fmt(statistics.stdev(xs)) if len(xs) > 1 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dirs", nargs="*", help="experiment directories (default: runs/sudoku_*)")
    ap.add_argument("--out", default="results/sudoku")
    ap.add_argument("--decoder", default="top_k_unmasking_1", help="primary decoder for targets and summary")
    ap.add_argument("--reeval-dirs", nargs="*", default=None, help="re-evaluation directories (default: runs/reeval_*)")
    ap.add_argument("--packages", nargs="*", default=None, help="W&B-style run packages (default: reports/sudoku_*/runs.json)")
    args = ap.parse_args()
    exp_dirs = args.exp_dirs or sorted(glob.glob("runs/sudoku_*"))
    os.makedirs(args.out, exist_ok=True)

    eval_rows, train_rows, run_rows, package_reeval_rows = [], [], [], []

    def add_run(exp, cond, seed, records, info, status, source):
        evals = [r for r in records if "val_loss" in r]
        key = dict(experiment=exp, condition=cond, condition_name=CONDITION_NAMES.get(cond, ""), seed=seed)
        for r in records:
            if "val_loss" in r:
                eval_rows.append({**key, "step": r["step"], "epoch": r["epoch"], "train_time_s": r["train_time_s"],
                                  "eval_time_s": r["eval_time_s"], "val_loss": r["val_loss"],
                                  **{f"acc_{d}": v for d, v in r["val_acc"].items()}})
            else:
                train_rows.append({**key, "step": r["step"], "epoch": r["epoch"], "loss": r["loss"],
                                   "lr": r["lr"], "train_time_s": r["train_time_s"]})
        row = {**key, "status": status, "source": source,
               "config": info.get("config", ""), "slurm_job": info.get("slurm_job", ""), "node": info.get("node", ""),
               "cuda_visible_devices": info.get("CUDA_VISIBLE_DEVICES", ""), "gpus": info.get("gpus", ""),
               "start": info.get("start", ""), "end": info.get("end", ""), "exit": info.get("exit", ""),
               "git_head": info.get("git_head", ""), "uncommitted_code_diff": info["uncommitted_code_diff"]}
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

    for exp_dir in exp_dirs:
        exp = os.path.basename(os.path.normpath(exp_dir))
        for run_dir in sorted(glob.glob(os.path.join(exp_dir, "*", "seed*"))):
            cond = os.path.basename(os.path.dirname(run_dir))
            seed = int(os.path.basename(run_dir)[len("seed"):])
            metrics = os.path.join(run_dir, "metrics.jsonl")
            if not os.path.exists(metrics):
                continue
            records = [json.loads(line) for line in open(metrics) if line.strip()]
            info = parse_run_info(os.path.join(run_dir, "run_info.txt"))
            code_diff = os.path.join(run_dir, "code.diff")
            info["uncommitted_code_diff"] = int(os.path.exists(code_diff) and os.path.getsize(code_diff) > 0)
            add_run(exp, cond, seed, records, info,
                    "done" if os.path.exists(os.path.join(run_dir, "DONE")) else "incomplete", exp_dir)

            raw_dir = os.path.join(args.out, "raw", exp, cond, f"seed{seed}")
            os.makedirs(raw_dir, exist_ok=True)
            for name in RAW_FILES:
                if os.path.exists(os.path.join(run_dir, name)):
                    shutil.copy2(os.path.join(run_dir, name), raw_dir)

    packages = args.packages if args.packages is not None else sorted(glob.glob("reports/sudoku_*/runs.json"))
    for package in packages:
        for exp, cond, seed, records, info, rescored in load_package_runs(package):
            final = [r for r in records if "val_loss" in r and r["step"] == max(x["step"] for x in records)]
            add_run(exp, cond, seed, records, info, "done" if final and info.get("exit") == "0" else "incomplete", package)
            package_reeval_rows += [{"reeval": package, "experiment": exp, "condition": cond,
                                     "condition_name": CONDITION_NAMES.get(cond, ""), "seed": seed, **r} for r in rescored]

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
             f"Generated by `scripts/export_sudoku_results.py` from {', '.join(f'`{d}`' for d in exp_dirs + packages)}.",
             f"Accuracy is exact-grid accuracy of the `{args.decoder}` decoder (a valid grid that keeps the givens); "
             "mean ± std over seeds.",
             "This table uses the bf16 eval logged during training; a target counts as reached at the first of two "
             "consecutive evals at or above it. The bf16 eval is unreliable for B, C and D (see the fp32 table below).",
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

    # re-scores of saved EMA checkpoints by scripts/reeval_sudoku.py (one directory per re-evaluation)
    reeval_rows = list(package_reeval_rows)
    for reeval_dir in args.reeval_dirs if args.reeval_dirs is not None else sorted(glob.glob("runs/reeval_*")):
        name = os.path.basename(os.path.normpath(reeval_dir))
        for path in sorted(glob.glob(os.path.join(reeval_dir, "shard*.jsonl"))):
            for line in open(path):
                r = json.loads(line)
                reeval_rows.append({"reeval": name, "experiment": r["experiment"], "condition": r["condition"],
                                    "condition_name": CONDITION_NAMES.get(r["condition"], ""), "seed": r["seed"],
                                    "step": r["step"], "final": int(r["final"]), "precision": r["precision"],
                                    "decoder": r["decoder"], "acc": r["acc"], "logged_bf16_acc": r["logged_bf16_acc"],
                                    "eval_time_s": r["eval_time_s"], "node": r["node"], "gpu": r["gpu"]})
        raw_dir = os.path.join(args.out, "raw", name)
        os.makedirs(raw_dir, exist_ok=True)
        for path in glob.glob(os.path.join(reeval_dir, "*.jsonl")) + glob.glob(os.path.join(reeval_dir, "run_info_job*.txt")) \
                + glob.glob(os.path.join(reeval_dir, "code_job*.diff")):
            shutil.copy2(path, raw_dir)
    if reeval_rows:
        reeval_rows.sort(key=lambda r: (r["reeval"], r["condition"], r["seed"], r["step"], r["precision"], r["decoder"]))
        write_csv("reeval.csv", reeval_rows)
        prim = [r for r in reeval_rows if r["decoder"] == args.decoder]
        repro = [abs(r["acc"] - r["logged_bf16_acc"]) for r in prim if r["precision"] == "bf16"]
        steps = sorted({r["step"] for r in prim if not r["final"]})
        lines += ["", "## fp32 re-score of saved EMA checkpoints",
                  "",
                  f"From `reeval.csv` (`scripts/reeval_sudoku.py`): the same eval with an fp32 decode. The bf16 eval used "
                  f"during training ranks cells by bf16 confidences, so near-1 confidences tie and are broken by position. "
                  f"bf16 re-runs reproduce the logged accuracy within {100 * max(repro):.2f} points "
                  f"({len(repro)} checkpoints)." if repro else "",
                  "",
                  "| Condition | Final acc. fp32 (%) | Final acc. bf16 as logged (%) | " + " | ".join(f"fp32 at {s // 1000}k" for s in steps) + " |",
                  "|---|---|---|" + "---|" * len(steps)]
        for cond in sorted({r["condition"] for r in prim}):
            fin = [r for r in prim if r["condition"] == cond and r["final"] and r["precision"] == "fp32"]
            cells = [f"{cond} ({CONDITION_NAMES.get(cond, '')})",
                     mean_std([100 * r["acc"] for r in fin], lambda x: f"{x:.1f}"),
                     mean_std([100 * r["logged_bf16_acc"] for r in fin], lambda x: f"{x:.1f}")]
            for s in steps:
                cells.append(mean_std([100 * r["acc"] for r in prim if r["condition"] == cond and r["step"] == s
                                       and r["precision"] == "fp32"], lambda x: f"{x:.1f}"))
            lines.append("| " + " | ".join(cells) + " |")
    open(os.path.join(args.out, "summary.md"), "w").write("\n".join(lines) + "\n")
    print(f"wrote {len(eval_rows)} eval rows, {len(train_rows)} train rows, {len(run_rows)} runs, "
          f"{len(reeval_rows)} re-eval rows to {args.out}")


if __name__ == "__main__":
    main()
