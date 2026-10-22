"""Import completed C results; default execution validates without network access.

Verified against the W&B Run.log/init documentation and SDK 0.30.0 signatures.
Each history step is merged before import; existing run IDs are never resumed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

SEEDS = (1000, 2000, 3000, 4000, 5000)
CHECKPOINTS = (10000, 20000, 30000, 40000, 50000, 53570)
PRIMARY = "acc_top_k_unmasking_1"


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Expected a version-1 import payload")
    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) != 5:
        raise ValueError("Expected exactly five C seed runs")
    observed = set()
    for run in runs:
        seed = run.get("seed")
        if isinstance(seed, bool) or seed not in SEEDS or seed in observed:
            raise ValueError(f"Invalid or duplicate seed: {seed}")
        observed.add(seed)
        if run.get("id") != f"C-seed{seed}-20261022":
            raise ValueError(f"Unexpected run ID for seed {seed}")
        config = run.get("config", {})
        if config.get("condition") != "C" or config.get("seed") != seed or config.get("main_eval") != "fp32":
            raise ValueError(f"Incorrect condition, seed, or main evaluation mode: {seed}")
        history = run.get("history")
        if not isinstance(history, list) or not history:
            raise ValueError(f"Empty or invalid history for seed {seed}")
        previous = -1
        bf16_steps, fp32_steps = [], []
        for row in history:
            step = row.get("optimizer_step")
            if isinstance(step, bool) or not isinstance(step, int) or not previous < step <= 53570:
                raise ValueError(f"History steps must increase strictly for seed {seed}")
            previous = step
            for key, value in row.items():
                if not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"History must contain finite scalar metrics: seed {seed}, {key}")
                if "/acc_" in key and not 0 <= value <= 1:
                    raise ValueError(f"Accuracy must use the 0-1 scale: seed {seed}, {key}")
            if f"eval_bf16/{PRIMARY}" in row:
                bf16_steps.append(step)
            has_fp32 = f"eval_fp32/{PRIMARY}" in row
            has_main = f"eval_main/{PRIMARY}" in row
            if has_fp32 != has_main or (has_fp32 and row[f"eval_fp32/{PRIMARY}"] != row[f"eval_main/{PRIMARY}"]):
                raise ValueError(f"eval_main must equal available FP32 observations: seed {seed}, step {step}")
            if has_fp32:
                fp32_steps.append(step)
        if bf16_steps != list(range(1000, 53570, 1000)) + [53570]:
            raise ValueError(f"Expected 54 BF16 evaluations for seed {seed}")
        if fp32_steps != list(CHECKPOINTS):
            raise ValueError(f"Expected exactly six FP32 observations for seed {seed}")
        if history[-1]["optimizer_step"] != 53570:
            raise ValueError(f"Missing final budget observation for seed {seed}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, default=Path(__file__).with_name("runs.json"))
    parser.add_argument("--entity", default="puma-mbz")
    parser.add_argument("--project", default="puma-sudoku")
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--upload", action="store_true", help="Create five runs in W&B; without this flag only validate")
    args = parser.parse_args()
    try:
        payload = load_payload(args.payload)
        if len(set(args.seeds)) != len(args.seeds):
            raise ValueError("Do not repeat a seed")
        selected = [r for r in payload["runs"] if r["seed"] in args.seeds]
        print(f"Validated {len(selected)} runs: 54 BF16 and 6 full-FP32 observations per seed.")
        if not args.upload:
            print("Validation only. No W&B connection or upload was performed.")
            return 0
        import wandb

        receipt_path = args.payload.parent / "upload_receipts.jsonl"
        for record in selected:
            with wandb.init(
                entity=args.entity, project=args.project, id=record["id"],
                name=record["name"], group="C", job_type="historical-import",
                config=record["config"], tags=record["tags"],
                mode="online", resume="never", save_code=False,
                settings=wandb.Settings(disable_git=True, x_disable_stats=True, console="off"),
            ) as run:
                run.define_metric("optimizer_step")
                run.define_metric("*", step_metric="optimizer_step")
                for row in record["history"]:
                    run.log(row, step=row["optimizer_step"], commit=True)
                run.summary.update(record["summary"])
                receipt = {"entity": args.entity, "project": args.project,
                           "seed": record["seed"], "run_id": run.id, "url": run.url}
            with receipt_path.open("a") as handle:
                handle.write(json.dumps(receipt) + "\n")
            print(f"Uploaded seed {record['seed']}: {receipt['url']}")
        return 0
    except Exception as error:
        print(f"Import stopped: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
