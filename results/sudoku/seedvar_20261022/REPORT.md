# Why some B and D seeds fail on Sudoku

Status: 2026-10-23. All sections rest on finished runs.

## 1. Answer

- **The failure is a downside of the B/D training regime, not an evaluation bug and not a bad initialisation.**
  B and D train only on boards with 100 / 75 / 51 / 26 % of the empty cells masked (schedule 81/61/41/21, rescaled), so
  never on fewer than ~14 masked cells. The greedy decoder always passes through boards with 3, 2 and 1 masked cells.
  On those boards some B/D models stop reading the board and output a fixed guess for the position. Which seeds do
  this, and how badly, varies with the seed and changes over training. Nothing in the loss constrains it.
- **The trigger is the total number of masked cells on the board, not the target cell's neighbourhood.** Adding
  masks that share no row, column or box with the target restores its accuracy (Sec. 5).
- **Training on a 1-mask state removes the failure.** 200–1000 extra updates with a fifth schedule state of 1 masked
  cell raise single-cell accuracy to 99.7–100 % and exact-grid accuracy from 46–74 % to 87–91 % on the four worst
  checkpoints. The same extra training with the original schedule does nothing, and the extra state does not change a
  good seed (Sec. 6).
- **It cannot be fixed inside B and D as the proposal defines them.** The defect sits in which boards the loss
  covers. Every fix changes either the training schedule (a new condition, which also changes B–F comparability) or
  the decoder (shared by all conditions). The cheapest valid fix is to extend the shared schedule by its next step,
  (81, 61, 41, 21, 1), for all of B–F. Full retrains with it: B 79.5 ± 17.5 -> 91.4 ± 3.0, D 74.1 ± 17.8 -> 90.9 ± 1.6 (Sec. 7).
- **Open:** why E, with the same 26 % floor, is immune. It was checked on one E checkpoint only.

## 2. Setup

Conditions as in the proposal (`IML_Project.pdf`, Table 1). Code: worktree `PUMA-team-BD`, branch `sudoku-b-d`, commit
`ebad40a` (fp32 ranking in the decoder). Model: 8 layers, width 256, 53 570 updates, 2 GPUs × batch 32, EMA 0.9999.
Evaluation: 999 test puzzles, greedy decoding with one cell per step by top confidence, exact-grid accuracy, fp32.
All probes use the final EMA checkpoints unless noted. Sec. 4 and 6 use the 20261019 B/D runs. Sec. 5 B rows and the
single-mask error analysis use the 20261022 B retrain (same code and seeds; B2000 51.1 vs 51.5 exact).

| condition | seeds | final exact acc, mean ± sd | per seed |
|---|---|---|---|
| A (Roman, bf16 eval) | 5 | 88.1 ± 1.8 | 89.7 85.7 88.1 87.2 89.9 |
| E (Roman, bf16 eval) | 5 | 95.4 ± 0.6 | 95.0 95.3 95.4 96.3 94.9 |
| B (fp32, `runs/sudoku_B_20261022`) | 10 | 81.7 ± 13.2 | 88.9 51.5 93.9 89.7 73.6 90.8 92.9 85.7 72.2 77.8 |
| D (fp32 rescore of `runs/sudoku_D_20261019`) | 5 | 74.1 ± 17.8 | 82.1 45.8 92.9 69.8 80.1 |

The bf16 eval does not distort E (bf16 95.0, fp32 95.8 on the rerun of seed 1000), so Roman's E numbers stand.

## 3. Ruled out

| candidate | test | result |
|---|---|---|
| bf16 ranking in the decoder | fp32 rescore, then fixed decoder (`ebad40a`) | fixed earlier; the spread above is already fp32 |
| EMA weights | B2000 single-mask accuracy, EMA vs raw | 84.9 % vs 84.6 %, same error pattern (`floor1/m1_errors.out`) |
| bad initial weights or data order | same seed under E (same init, same sampler seed) | E seed 2000 95.3, seed 5000 94.9 (Roman) vs B 51.5/73.6, D 45.8/80.1 |
| environment and data | E seed 1000 rerun in our env | identical to Roman's at every eval (feature file, 2026-10-21) |

## 4. Where the errors are

fp32 greedy decode, number of puzzles by masked cells left at the first wrong reveal (`probe_final.jsonl`, 999 puzzles):

| ckpt | exact | 1 left | 2–3 | 4–8 | ≥9 | single-mask acc |
|---|---|---|---|---|---|---|
| B2000 | 51.1 | 291 | 84 | 27 | 87 | 86.6 |
| D2000 | 45.8 | 227 | 127 | 58 | 129 | 82.9 |
| B5000 | 73.8 | 141 | 26 | 11 | 84 | 96.1 |
| B3000 (good) | 93.7 | 0 | 0 | 0 | 63 | 100 |
| E seed 1000 | 95.8 | 1 | 0 | 0 | 41 | 100 |

Re-masking random empty cells of solved grids (one forward pass): accuracy is ≈100 % for every B/D checkpoint from 8 masked
cells up to the training range, and falls as the count drops to 1 (B2000: 99.99 / 99.4 / 97.3 / 93.4 / 86.6 % at 8/5/3/2/1).

What the wrong answers look like (B2000, every empty cell of 200 puzzles masked on its own, `floor1/m1_errors.out`):
- 100 % of wrong digits already occur in the cell's row, column or box, so the model is not using the constraints.
- They depend on position (row 8: 50 % wrong, row 5: 4 %) and digit (wrong "2": 621 times, wrong "8": never).

## 5. Total masked count, not local context

One target empty cell per puzzle. The other m−1 masks go on empty cells that share no row/col/box with the target
("far") or on its peers first ("near"). Accuracy on the target cell, 999 puzzles (`floor1/ctx_probe.out`):

| ckpt | m=1 | far 2 | far 4 | far 8 | near 2 | near 4 | near 8 |
|---|---|---|---|---|---|---|---|
| B2000 | 86.6 | 94.1 | 99.6 | 100.0 | 91.6 | 97.2 | 99.4 |
| D2000 | 82.3 | 91.4 | 99.1 | 99.9 | 88.5 | 95.7 | 99.4 |
| B5000 | 96.2 | 98.8 | 99.9 | 100.0 | 99.1 | 99.9 | 100.0 |
| E seed 1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

The information available to the target is the same in "m=1" and "far m" (the extra masks are irrelevant cells), yet
3 extra irrelevant masks fix B2000. So the weak models need several masks to be present to work at all. That is a
property of what they were trained on (always ≥ ~14 masks), not of the reasoning the cell requires. It also explains
why the drop starts at ~5 masks rather than at 14: a few masks are already close enough to the trained range.

## 6. Causal test: fine-tune with and without a 1-mask state

Each final checkpoint is trained further with its own condition's pool code (B: fresh examples, D: nested random
reveals). Batch 64, AdamW lr 3e-5 constant, fp32, seed 0. The two arms differ only in the schedule:
sched4 = (81, 61, 41, 21) (control for extra training), sched5 = (81, 61, 41, 21, 1). Decode as above
(`floor1/finetune_floor.py`, `floor1/finetune_grid.jsonl`, CPU job 336623).

Exact-grid accuracy % (single-mask accuracy %):

| ckpt | before | sched4, 200 upd. | sched5, 200 upd. | sched4, 1000 upd. | sched5, 1000 upd. |
|---|---|---|---|---|---|
| B2000 | 51.1 (86.6) | 49.2 (85.8) | 88.2 (99.8) | 45.3 (85.7) | 89.3 (99.9) |
| D2000 | 45.8 (82.9) | 43.9 (80.8) | 85.6 (99.8) | 43.7 (81.5) | 87.4 (99.8) |
| B5000 | 73.8 (96.1) | 73.5 (96.2) | 91.0 (100) | 70.1 (95.7) | 90.8 (100) |
| D4000 | 69.8 (92.5) | 65.7 (92.1) | 86.7 (99.7) | 66.6 (92.1) | 88.0 (100) |
| B3000 (good) | 93.7 (100) | 92.5 (100) | 92.0 (100) | 93.0 (100) | 92.5 (100) |

- Late failures (first error with 1–3 cells left) drop from 179–420 to 0–27 per 999 puzzles with sched5; with sched4
  they stay.
- Even 20 updates help: B2000 48.1 → 71.3.
- What remains after the fix is the early-error rate (first error with ≥4 cells left): 97–127 per 999 for repaired
  B/D, 70–80 for B3000, 41 for E. So repaired B/D land at 86–91 %, near the good B seeds (88.9–93.9) but below E.

## 7. Full retrains with the extended schedule

B5 and D5 = B and D with `mask_schedule: [81, 61, 41, 21, 1]`, everything else identical (same seeds, so same initial
weights and data order). Jobs 336849 (B5), 336850 (D5); raw outputs `runs/sudoku_{B5,D5}_20261022/seed<s>/`; final
`top_k_unmasking_1` as logged (fp32 ranking, commit `ebad40a` + untracked configs). B and D columns: the 20261022
retrain logs (same code).

| seed | B | B5 | D | D5 |
|---|---|---|---|---|
| 1000 | 88.9 | 86.8 | 83.3 | 88.7 |
| 2000 | 51.5 | 93.8 | 46.2 | 90.5 |
| 3000 | 93.9 | 90.2 | 92.7 | 92.8 |
| 4000 | 89.7 | 93.8 | 69.0 | 90.3 |
| 5000 | 73.6 | 92.6 | 79.5 | 92.0 |
| mean ± sd | 79.5 ± 17.5 | 91.4 ± 3.0 | 74.1 ± 17.8 | 90.9 ± 1.6 |

Mask-level probe and fp32 decode on the B5/D5 final EMA checkpoints (`probe_mask_level.py`, local CPU run,
`floor1/probe_floor1.jsonl`; all except D5 seed 4000, still training at probe time): single-mask accuracy 100.0 % for
all 9 (also at 2, 3, 8 masks); puzzles whose first error comes with 1–3 cells left: 1 in total over the 9 models
(B5 3000), vs 19–375 per B/D seed before; first errors with 4–15 cells left: 0–8 per model. What remains is early
errors (first error with ≥16 cells left), as for E and B3000.

The low outliers disappear (minimum 86.8 vs 46.2) and the sd falls to the range of Roman's runs (A 1.8, F 1.3, P 1.2,
PN 1.6; E 0.6). Good seeds move a few points either way (B1000 −2.1, B3000 −3.7). B5/D5 stay below E (95.4).

## 8. Is this the training regime?

Yes, in the narrow sense that matters for the study:
- The B/D regime never trains below ~14 masks; the evaluation protocol always visits 1–3. Fixing exactly that gap
  (Sec. 6) removes the failure, and nothing else tried does.
- A, which draws 1–81 masks, shows no such failure (sd 1.8).

But the floor alone is not sufficient: E and F use the same schedule, so the same 26 % floor, and are stable (sd 0.6 and 1.3).
PUMA and PUMA-without-collapse sample ratios across the whole range down to 1 mask, so they do train on near-empty boards.
The difference between D and E is only the reveal rule (random vs confidence), so random selection of the remaining
cells makes the models depend on the mask count, and confidence selection doesn't. Why is not established. Only one E
checkpoint was probed (seed 1000).

Consequences for the study:
- B and D numbers as defined are high-variance because of a protocol gap (trained mask range ≠ decoded range), not
  because of what B and D are meant to test (schedule, reuse, nesting, reveal rule). Their mean ± sd should be
  reported with this explanation, or the schedule should be extended for all of B–F.
- Extending only B and D would make A–B and D–E comparisons differ in the schedule as well. If the extended schedule
  is adopted, E (and F) need it too (`sudoku_E5_floor1.yaml` is prepared, not run).

## 9. Reproduce

All paths relative to `/home/timur.aisin/intro_to_ml`. Python: `miniconda3/envs/puma/bin/python`. Code: `PUMA-team-BD`
at `ebad40a` plus the untracked variant configs and launcher (copies in `floor1/`).

```sh
# Sec. 4 probe (CPU job 334771)
sbatch PUMA-team/results/sudoku/seedvar_20261022/probe.sbatch
# Sec. 4 single-mask errors and Sec. 5 global-vs-local probe (a few CPU minutes each)
python -I floor1/m1_errors.py runs/sudoku_B_20261022/seed2000/ema_step=53570.pt runs/sudoku_B_20261022/seed2000/step=53570.pt runs/sudoku_B_20261022/seed3000/ema_step=53570.pt
python -I floor1/ctx_probe.py runs/sudoku_B_20261022/seed2000/ema_step=53570.pt runs/sudoku_D_20261019/seed2000/ema_step=53570.pt runs/sudoku_B_20261022/seed5000/ema_step=53570.pt runs/sudoku_Erepro_20261019/seed1000/ema_step=53570.pt
# Sec. 6 fine-tune grid (CPU, ~1.6 h on 64 cores)
sbatch floor1/finetune_grid.sbatch
# Sec. 7 retrains (from PUMA-team-BD)
COND=B5 CFG=yaml_files/sudoku_B5_floor1.yaml EXP_DIR=$PWD/../runs/sudoku_B5_20261022 PY=... \
  sbatch --qos=cscc-gpu-qos --time=1-00:00:00 scripts/run_sudoku_variant_debug.sbatch 2000 5000 1000 3000 4000
```

`floor1/pool5_check.py` checks on synthetic data that the 5-state schedule produces exactly 1 masked cell in its last
state for B, D and E.
