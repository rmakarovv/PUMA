# Sudoku results (for analysis)

Exported from the cluster run directories by `scripts/export_sudoku_results.py`. Checkpoints are
not in git; they remain under `runs/` on the cluster (`/home/roman.makarov/project/PUMA/runs/`).
To regenerate after new runs, from the repo root:
```bash
python scripts/export_sudoku_results.py            # all runs/sudoku_* experiments -> results/sudoku
```
`summary.md` holds the mean ± std table per condition.

## Conditions (see `docs/sudoku_AE.md`)
| Code | Name | Config | Collapse (confidence fast-forwarding) |
|---|---|---|---|
| A | random masking | `yaml_files/sudoku_A_random.yaml` | n/a |
| E | confidence progression | `yaml_files/sudoku_E_confidence.yaml` | no (`mode: standard`) |
| F | Gt-Margin progression | `yaml_files/sudoku_F_gtmargin.yaml` | no (`mode: standard`) |
| P | PUMA reference (stock) | `yaml_files/sudoku_PUMA_reference.yaml` | yes (`mode: confidence_collapse`, τ = 0.9, K = 10) |
| PN | PUMA without collapse | `yaml_files/sudoku_PUMA_nocollapse.yaml` | no (`mode: standard`, K = 10) |

All runs use seeds 1000–5000 and 53,570 updates (batch 32 × 2 GPUs). Runs with the same seed
start from identical weights.

## Evaluation protocol
Evaluation runs every 1000 updates and at the final step, with EMA weights, on the first 999
puzzles of the test set. Accuracy is exact-grid accuracy for six greedy decoders
`{top_k, top_k_margin}_unmasking_{1,2,3}`: confidence or top-2 margin ranking, filling 1–3 cells
per step. The proposal's decoder is `top_k_unmasking_1`. `val_loss` is the random-mask MDM loss
on the 5% held-out training split, computed with non-EMA weights.

## Files
- `evals.csv`: one row per evaluation. Columns are `experiment, condition, condition_name, seed,
  step, epoch, train_time_s, eval_time_s, val_loss`, plus `acc_<decoder>` for each of the six
  decoders. `train_time_s` is cumulative wall time excluding evaluation.
- `train_log.csv`: one row per 100 training steps (`step, epoch, loss, lr, train_time_s`).
  `loss` is the training minibatch loss on rank 0 at that step.
- `runs.csv`: one row per run.
  - Provenance: `status, config, slurm_job, node, cuda_visible_devices, gpus, start, end, exit,
    git_head, uncommitted_code_diff`. `gpus` lists every GPU nvidia-smi showed on the node; the
    allocated ones are given by `cuda_visible_devices`.
  - Final metrics: `final_step, final_val_loss, train_time_min, eval_time_min, final_acc_<decoder>`,
    and `best_acc` (the primary decoder's best eval).
  - Targets: `updates_to_{50,75,90}` and `train_min_to_{50,75,90}`, using the primary decoder.
    A target counts as reached at the first of two consecutive evals at or above it; the cell is
    empty if the run never reached it.
- `reeval.csv`: re-scores of the saved EMA checkpoints (every 10k updates and the final step) by
  `scripts/reeval_sudoku.py`, one row per checkpoint, precision and decoder. `precision` is `fp32` (fp32
  decode) or `bf16` (the original eval, final checkpoints only). `logged_bf16_acc` is the value logged
  during training. The bf16 eval ranks cells by bf16 confidences, so near-1 values tie and are filled in
  grid order. Raw outputs and job records are in `raw/reeval_*`.
- `raw/<experiment>/<condition>/seed<seed>/`: the original `metrics.jsonl`, `run_info.txt`,
  `config.yaml` (resolved config) and `code.diff`. `code.diff` holds uncommitted changes relative
  to `git_head` at launch, and is empty when the tree was clean.
