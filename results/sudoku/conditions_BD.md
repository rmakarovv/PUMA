# Sudoku conditions B and D, and the bf16 eval issue

## Conditions
| Code | Name | Config | What changes vs E |
|---|---|---|---|
| B | schedule only | `yaml_files/sudoku_B_schedule.yaml` | `reuse_examples: false`: fresh example and independent random mask every update (same mask ladder) |
| D | nested, random reveal | `yaml_files/sudoku_D_random.yaml` | `reveal_score: random`: each stage reveals a uniformly random subset of the masked cells |

Both use the E setup otherwise (ladder [81,61,41,21] rescaled to L_eff, `mode: standard`,
batch 32 x 2 GPUs, 53,570 updates, seeds 1000-5000). Launchers: `scripts/run_sudoku_B.sbatch`,
`scripts/run_sudoku_D.sbatch` (output `$EXP_DIR/seed<s>/`, no condition subfolder).
Env: conda `puma`, torch 2.11.0+cu128.

`raw/sudoku_Erepro_20261019/seed1000` is E seed 1000 re-run with our env and data cache; it
matches Roman's `raw/sudoku_AE_20261016/E/seed1000` exactly at every eval (setup check).

## bf16 eval issue
`sampling.mdm_sampling` ranks masked cells by `F.softmax(logits)` with bf16 logits, so the
confidences are bf16. Values >= ~0.998 all round to 1.0 (bf16 spacing below 1 is 2^-8), about 90%
of decode steps then have a tied top score, and `torch.topk` returns the lowest tied index (CPU and
CUDA). Decoding becomes top-left first among near-equal confidences. E/F/P models are insensitive to
this order; B/D models fail at the last few cells. All logged accuracies (ours and Roman's) use this
eval.

Ablation (`rescore_fp32_20261021/abl_*`, CPU, 300 puzzles, final checkpoints, exact-grid %):

| ranking | E (seed 1000) | D seed 1000 | B seed 2000 |
|---|---|---|---|
| fp32 | 95.3 | 83.7 | 50.0 |
| bf16-rounded, lowest index on ties (= current eval) | 94.3 | 37.3 | 4.7 |
| bf16-rounded, highest index on ties | 94.3 | 89.3 | 70.7 |
| bf16-rounded, random tie-break | 95.7 | 86.0 | 60.7 |
| bf16 forward, fp32 softmax | 95.3 | 85.0 | 48.3 |

Fix (verified on GPU through the repo eval, `fixcheck_330027.out`; applied in `sampling.py` on this branch, re-checked in `check_patch_334566.out`):
`logits = logits.float()` right after the bf16 forward in `sampling.mdm_sampling` (and `mdm_sampling_block`), so softmax, ranking and argmax run in fp32. On all 11 final checkpoints it
is within 1.2 points of a full fp32 decode.

## Results (final step, top_k_unmasking_1, exact-grid accuracy, 999 puzzles)
| Cond. | logged (bf16 eval) | fp32 re-score | per seed fp32 (1000..5000) |
|---|---|---|---|
| B | 62.2 ± 37.2 | 79.4 ± 17.6 | 89.5 / 51.1 / 93.7 / 89.0 / 73.8 |
| D | 53.9 ± 37.6 | 74.1 ± 17.8 | 82.1 / 45.8 / 92.9 / 69.8 / 80.1 |
| E (seed 1000 re-run) | 95.0 | 95.8 | 95.8 |

Roman's A/F/P/PN checkpoints were not re-scored (not accessible), so his numbers in `evals.csv`
remain bf16-eval numbers. fp32 re-scores exist only at the saved checkpoints (every 10k updates).
Some seeds still decline late in fp32 (B 2000: 64.4 at 30k -> 51.1; D 2000: 55.3 -> 45.8;
D 4000: 76.3 -> 69.8).

## Files
- `raw/sudoku_{B,D}_20261019/seed<s>/`, `raw/sudoku_Erepro_20261019/seed1000/`: `metrics.jsonl`,
  `config.yaml`, `run_info.txt`, `code.diff` per run (checkpoints stay on the cluster under
  `/home/timur.aisin/intro_to_ml/runs/`).
- `rescore_fp32_20261021/<COND>_seed<s>.jsonl`: per checkpoint, the six decoders in fp32 plus
  `top_k_unmasking_1` in bf16 (matches the logged value). Produced by `rescore.py`/`rescore.sbatch`
  (repo eval code unchanged, model wrapped to run without autocast).
- `rescore_fp32_20261021/`: `ablate_bf16.py` (table above), `trace_order.py` (where the first error
  happens per tie order), `fixcheck.py` (GPU check of the fix and of CUDA `topk` tie order),
  `decode_diag.py` (per-cell accuracy and error histograms), with their outputs. Scripts contain the
  cluster paths they were run with.
