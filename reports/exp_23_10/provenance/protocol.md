# Condition C: schedule-length pilot

This is an exploratory follow-up to the completed five-seed C study. It uses
new output directories and a frozen source snapshot. The original study and
its published result package remain the baseline.

## Prespecified design

| Variant | Nominal mask-count schedule | Seeds | Maximum updates |
|---|---|---|---:|
| C-K2 | 81, 41 | 1000, 2000, 3000, 4000, 5000 | 30,000 |
| C-K4 | 81, 61, 41, 21 | Same five seeds | 30,000 |
| C-K8 | 81, 71, 61, 51, 41, 31, 21, 11 | Same five seeds | 30,000 |

The schedules are rescaled to each puzzle's empty cells. This changes both
example lifetime and mask-count distribution; it does not isolate reuse alone.
All variants use independent masks at each state, staggered batch stages,
visible givens, and teacher-forced visible solution cells.

Architecture, batch size (32/rank on two GPUs), AdamW, learning rate (3e-4),
warmup (1,000), weight decay, EMA and gradient clipping retain the original
C settings. Training uses BF16 mixed precision and FP32 weights. The cosine
schedule retains its original 53,570-update horizon; training stops at 30,000.
Thus changing the pilot budget does not compress the learning-rate schedule.
Each paired seed starts from identical weights, recorded by SHA256. Equal
updates imply different numbers of new example draws and supervised tokens
across the schedules. The inherited pool's data-loader consumption is retained.

## Validation and selection

The fixed generation set contains 1,024 distinct puzzles from the original 5%
training holdout (split seed 2026). Puzzle duplicates appearing anywhere in the
training subset are excluded before selection. The supplied test set is not
used for pilot generation or parameter selection. A data manifest records
selection indices, source hashes and overlap counts.

Generation uses EMA weights, greedy `top_k_unmasking_1`, FP32 forward/softmax,
TF32 disabled, and 16 puzzles per rank per evaluation batch. Evaluate and save
at updates 5,000, 10,000, 15,000, 20,000, 25,000 and 30,000. The metric is a valid
completed Sudoku preserving givens. Held-out random-mask loss uses the same
1,024 puzzles, non-EMA weights, the inherited BF16 loss computation, and fixed
mask RNG seed 4601 plus rank. Evaluation restores training RNG and matmul settings.

The primary endpoint is mean final validation accuracy at update 30,000 with
sample SD (`ddof=1`), reporting all five seeds. Paired differences versus C-K4
and full curves are secondary diagnostics. Partial results remain provisional.
No best-seed or best-checkpoint substitution is made. A promising variant
requires a later matched-budget comparison against F; this pilot cannot
establish superiority over F or be compared directly to the old test scores.

## Execution

Run `scripts/prepare_sudoku_C_pilot.sbatch` on a CPU node with a new `EXP_DIR`.
It runs the holdout, precision and existing C tests, prepares data, freezes
source/configuration files and writes the complete 15-run plan.

Then run `scripts/smoke_sudoku_C_pilot.sbatch` with `afterok` on preparation.
The smoke check trains all three schedules for 24 updates on two GPUs and
checks final checkpoints, stopping behavior, FP32 evaluation and LR horizon.
Submit the two-worker array `scripts/run_sudoku_C_pilot.sbatch` only after this
passes. Each worker requests two GPUs and processes its assigned frozen runs
sequentially, respecting the account's two-job/four-GPU allowance.

Schedule `scripts/finalize_sudoku_C_pilot.sbatch` with `afterany` on the training
array. Its analysis directory contains per-seed scores, six-point curves,
paired differences, a mean/SD table and a figure. Failures remain explicit.
All launchers use the existing `.venv`; new pilot metrics stay local until a
separate W&B import is requested.

## External verification

- [PyTorch autocast documentation](https://docs.pytorch.org/docs/2.9/amp.html)
- [Current autocast implementation](https://github.com/pytorch/pytorch/blob/main/torch/amp/autocast_mode.py)
- [Top-k tie behavior](https://docs.pytorch.org/docs/2.9/generated/torch.topk.html)
- [JIT/autocast issue 75956](https://github.com/pytorch/pytorch/issues/75956), reviewed for context; this pilot uses eager execution, not JIT.
- [Slurm job-array documentation](https://slurm.schedmd.com/job_array.html)
