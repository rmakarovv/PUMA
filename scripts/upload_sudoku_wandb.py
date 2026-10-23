"""Upload exported Sudoku run records to wandb, one wandb run per (condition, seed).

Reads results/sudoku/raw/<exp>/[<COND>/]seed<S>/ (metrics.jsonl, run_info.txt, config.yaml) and, where present,
the fp32 re-scores results/sudoku/rescore_fp32_20261021/<COND>_seed<S>.jsonl. Logged keys, all on the training step:
  train/loss, train/lr          every 100 updates
  eval/val_loss                 every eval (random-mask MDM loss, non-EMA weights)
  eval_bf16/acc_<decoder>       every eval, as logged during training (bf16 decode ranking)
                                (runs in FP32_EVAL_EXPERIMENTS ranked in fp32 during training: logged as eval_fp32)
  eval_fp32/acc_<decoder>       fp32 re-score of the saved EMA checkpoints (10k..final steps only)
  eval_main/acc_<decoder>       eval_fp32 where the run was re-scored, else eval_bf16 (config key main_eval says which)
The wandb group is the condition, so the UI can aggregate seeds. Run ids are fixed (<COND>-seed<S>-<id-suffix>), so uploading
the same run twice fails instead of duplicating it. wandb never reuses a deleted run's id: to re-upload, delete the
runs and pass a new --id-suffix.
Usage: python scripts/upload_sudoku_wandb.py [--entity puma-mbz] [--project puma-sudoku] [--only B D] [--dry-run]
"""
import argparse
import glob
import json
import os
import re

import yaml

# experiments trained after the eval fix (ebad40a): the logged accuracy is already fp32
FP32_EVAL_EXPERIMENTS = {"sudoku_B_20261022", "sudoku_D_20261022"}

CONDITION_NAMES = {
    "A": "random masking",
    "B": "shared schedule, fresh examples",
    "D": "reused, nested, random reveal",
    "E": "confidence progression",
    "Erepro": "confidence progression (rerun, our env)",
    "F": "Gt-Margin progression",
    "P": "PUMA reference",
    "PN": "PUMA without collapse",
}


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def find_runs(raw_dir, rescore_dir):
    runs = []
    for metrics in sorted(glob.glob(os.path.join(raw_dir, "*", "**", "metrics.jsonl"), recursive=True)):
        run_dir = os.path.dirname(metrics)
        exp = os.path.relpath(run_dir, raw_dir).split(os.sep)[0]
        seed = int(os.path.basename(run_dir)[len("seed"):])
        info_text = open(os.path.join(run_dir, "run_info.txt")).read()
        cond = re.search(r"condition=(\S+)", info_text).group(1)
        cond = re.sub(r"\W", "", cond)  # "E(repro)" -> "Erepro"
        info = dict(re.findall(r"(?:^|\s)(slurm_job|node|git_head|start|end|exit)=(\S*)", info_text))
        rescore = os.path.join(rescore_dir, f"{cond}_seed{seed}.jsonl")
        if exp in FP32_EVAL_EXPERIMENTS:  # the rescores belong to the earlier runs
            rescore = ""
        runs.append(dict(exp=exp, cond=cond, seed=seed, run_dir=run_dir, info=info,
                         rescore=rescore if os.path.exists(rescore) else None, fp32_eval=exp in FP32_EVAL_EXPERIMENTS))
    return runs


def build_history(run):
    """Merge train, eval and fp32 rows into {step: {key: value}}, so each step is logged once and in order."""
    hist = {}
    for rec in read_jsonl(os.path.join(run["run_dir"], "metrics.jsonl")):
        row = hist.setdefault(rec["step"], {"epoch": rec["epoch"]})
        if "loss" in rec:
            row["train/loss"] = rec["loss"]
            row["train/lr"] = rec["lr"]
            row["train/time_s"] = rec["train_time_s"]
        if "val_loss" in rec:
            row["eval/val_loss"] = rec["val_loss"]
            row["eval/time_s"] = rec["eval_time_s"]
            prefix = "eval_fp32/" if run["fp32_eval"] else "eval_bf16/"
            for dec, acc in rec["val_acc"].items():
                row[f"{prefix}acc_{dec}"] = acc
    if run["rescore"]:
        for rec in read_jsonl(run["rescore"]):
            row = hist.setdefault(rec["step"], {})
            for dec, acc in rec["fp32"].items():
                row[f"eval_fp32/acc_{dec}"] = acc
    main = "eval_fp32/" if run["rescore"] or run["fp32_eval"] else "eval_bf16/"
    for row in hist.values():
        row.update({"eval_main/" + k[len(main):]: v for k, v in list(row.items()) if k.startswith(main)})
    return hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entity", default="puma-mbz")
    p.add_argument("--project", default="puma-sudoku")
    p.add_argument("--raw", default="results/sudoku/raw")
    p.add_argument("--rescore", default="results/sudoku/rescore_fp32_20261021")
    p.add_argument("--only", nargs="*", help="conditions to upload (default: all)")
    p.add_argument("--exp", nargs="*", help="experiments to upload, e.g. sudoku_B_20261022 (default: all)")
    p.add_argument("--id-suffix", default="v2")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    runs = find_runs(args.raw, args.rescore)
    if args.only:
        runs = [r for r in runs if r["cond"] in args.only]
    if args.exp:
        runs = [r for r in runs if r["exp"] in args.exp]
    for run in runs:
        hist = build_history(run)
        n_fp32 = sum(any(k.startswith("eval_fp32/") for k in row) for row in hist.values())
        print(f"{run['cond']:>6} seed{run['seed']}  {run['exp']}  steps={len(hist)}  fp32_evals={n_fp32}")
        if args.dry_run:
            continue
        import wandb

        with open(os.path.join(run["run_dir"], "config.yaml")) as f:
            config = yaml.safe_load(f)
        config.update(condition=run["cond"], condition_name=CONDITION_NAMES.get(run["cond"], ""), seed=run["seed"],
                      experiment=run["exp"], fp32_rescored=run["rescore"] is not None,
                      main_eval="fp32" if run["rescore"] or run["fp32_eval"] else "bf16", fp32_eval_in_training=run["fp32_eval"], **run["info"])
        wb = wandb.init(entity=args.entity, project=args.project, id=f"{run['cond']}-seed{run['seed']}-{args.id_suffix}",
                        name=f"{run['cond']}-seed{run['seed']}", group=run["cond"], tags=[run["exp"]],
                        config=config, resume="never")
        for step in sorted(hist):
            wb.log(hist[step], step=step)
        wb.finish()


if __name__ == "__main__":
    main()
