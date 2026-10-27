"""Re-evaluate saved Sudoku EMA checkpoints (ema_step=*.pt) with a chosen decode precision.

Uses the training-time evaluation (eval/sudoku_eval.py: first 1% of the test set, six greedy
decoders) on one GPU per process. Checkpoints are split into disjoint shards; each process appends
one JSON line per (checkpoint, precision, decoder) to <out-dir>/shard<i>.jsonl and skips entries
already there, so an interrupted job can be resubmitted. Final checkpoints are evaluated first.
Usage: python scripts/reeval_sudoku.py --out-dir runs/<name> --shard 0 --num-shards 4 --gpu 0
"""
import argparse
import glob
import json
import os
import re
import socket
import sys
import time

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.transformer import MDMTransformer, MDMConfig
from eval.sudoku_eval import evaluate_ddp_sudoku

DECODERS = [(conf, k) for conf in ("top_k", "top_k_margin") for k in (1, 2, 3)]


def list_checkpoints(exp_dirs):
    """All EMA checkpoints as (path, experiment, condition, seed, step, is_final), finals first."""
    items = []
    for exp_dir in exp_dirs:
        for run_dir in sorted(glob.glob(os.path.join(exp_dir, "*", "seed*"))):
            paths = glob.glob(os.path.join(run_dir, "ema_step=*.pt"))
            steps = {p: int(re.search(r"ema_step=(\d+)\.pt$", p).group(1)) for p in paths}
            for p, step in steps.items():
                items.append((p, os.path.basename(os.path.normpath(exp_dir)), os.path.basename(os.path.dirname(run_dir)),
                              int(os.path.basename(run_dir)[len("seed"):]), step, step == max(steps.values())))
    return sorted(items, key=lambda it: (not it[5], -it[4], it[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--exp-dirs", nargs="*", default=None, help="default: runs/sudoku_*")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0, help="logical GPU index within the allocation")
    ap.add_argument("--precisions", nargs="+", default=["fp32"])
    ap.add_argument("--bf16-final", action="store_true", help="also evaluate final checkpoints with the bf16 forward")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    items = list_checkpoints(args.exp_dirs or sorted(glob.glob("runs/sudoku_*")))[args.shard::args.num_shards]
    out_path = os.path.join(args.out_dir, f"shard{args.shard}.jsonl")
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            r = json.loads(line)
            done.add((r["checkpoint"], r["precision"], r["decoder"]))
    plan = [(it, prec) for it in items for prec in args.precisions + (["bf16"] if args.bf16_final and it[5] and "bf16" not in args.precisions else [])]
    print(f"shard {args.shard}/{args.num_shards}: {len(items)} checkpoints, {len(plan)} (checkpoint, precision) passes, "
          f"{len(done)} decoder results already in {out_path}", flush=True)
    if args.dry_run:
        for (path, exp, cond, seed, step, final), prec in plan:
            print(f"  {prec} {path}{' (final)' if final else ''}")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    gpu_name = torch.cuda.get_device_name(device)
    loaded = None
    for (path, exp, cond, seed, step, final), prec in plan:
        todo = [(c, k) for c, k in DECODERS if (path, prec, f"{c}_unmasking_{k}") not in done]
        if not todo:
            continue
        if loaded != path:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            assert ckpt["is_ema_snapshot"] and ckpt["global_step"] == step, path
            cfg = OmegaConf.create(ckpt["config"])
            model = MDMTransformer(MDMConfig(**cfg.model)).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            loaded = path
        for conf, k in todo:
            decoder = f"{conf}_unmasking_{k}"
            sampling = OmegaConf.create({"temperature": 0.0, "confidence": conf, "unmasking_num": k, "precision": prec})
            t0 = time.time()
            with torch.inference_mode():
                acc = evaluate_ddp_sudoku(model, cfg, device, 0, 1, sampling)
            row = dict(experiment=exp, condition=cond, seed=seed, step=step, final=final, checkpoint=path,
                       precision=prec, decoder=decoder, acc=acc, logged_bf16_acc=ckpt.get(decoder),
                       eval_time_s=round(time.time() - t0, 2), node=socket.gethostname(), gpu=gpu_name)
            with open(out_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{prec} {exp}/{cond}/seed{seed} step {step} {decoder}: {acc:.4f} (logged bf16 {row['logged_bf16_acc']})", flush=True)


if __name__ == "__main__":
    main()
