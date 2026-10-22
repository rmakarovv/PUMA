# Condition C comparison

Budget: 53,570 updates. Decoder: `top_k_unmasking_1`. Seeds: 1000, 2000, 3000, 4000, 5000.
Training-curve comparisons use legacy bf16 confidence softmax. Values are mean ± sample SD; final columns include completed seeds only.

BF16 confidence rounding can change reveal order and materially affect accuracy and threshold times. These legacy results describe the original numerical decoder. A/F full-fp32 re-scores are unavailable here, so the separate C re-scores cannot establish an A/C/F ranking under full-fp32 evaluation.

| Condition | Completed / requested | Final accuracy (%) | Final validation loss | Train minutes | Eval minutes | Updates to 50% | Updates to 75% | Updates to 90% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 5 / 5 | 88.11 ± 1.76 | 0.0810 ± 0.0078 | 32.33 ± 0.34 | 65.14 ± 1.30 | 7400 ± 548 | 16000 ± 2000 | 2 / 5 confirmed |
| C | 5 / 5 | 69.51 ± 24.30 | 0.0760 ± 0.0052 | 33.01 ± 0.40 | 64.89 ± 0.95 | 8800 ± 3421 | 4 / 5 confirmed | 1 / 5 confirmed |
| F | 5 / 5 | 95.74 ± 1.26 | 0.0781 ± 0.0080 | 33.50 ± 0.42 | 66.22 ± 0.46 | 5200 ± 447 | 11200 ± 1643 | 22600 ± 5273 |

Paired final differences use C minus the baseline; positive accuracy differences favor C. Negative update differences indicate fewer C updates.

| Contrast | Completed pairs | Final accuracy difference (percentage points) |
|---|---:|---:|
| C-A | 5 / 5 | -18.60 ± 25.70 |
| C-F | 5 / 5 | -26.23 ± 23.69 |

## Separate C checkpoint re-scores

These values do not replace legacy bf16 results or establish new threshold times.

| Precision | Checkpoint update | Seeds | Accuracy (%) |
|---|---:|---:|---:|
| bf16_fp32_softmax | 53570 | 5 / 5 | 82.64 ± 10.77 |
| full_fp32 | 10000 | 5 / 5 | 62.16 ± 10.39 |
| full_fp32 | 20000 | 5 / 5 | 79.18 ± 10.32 |
| full_fp32 | 30000 | 5 / 5 | 83.86 ± 10.04 |
| full_fp32 | 40000 | 5 / 5 | 84.70 ± 9.22 |
| full_fp32 | 50000 | 5 / 5 | 83.12 ± 10.64 |
| full_fp32 | 53570 | 5 / 5 | 82.48 ± 10.90 |
| legacy_bf16 | 53570 | 5 / 5 | 69.51 ± 24.30 |

`per_seed.csv` includes partial-run status, last observed scores, confirmation updates, and measured timing. `paired.csv` contains per-seed differences.
A crossing needs two consecutive scheduled evaluations at or above the target; it is assigned to the first. A final unconfirmed crossing is not reached. Missing intermediate evaluations do not confirm a crossing.
Generation scores check completed Sudoku validity and preservation of givens. Training and evaluation times are separate; wall time also depends on hardware and launch/checkpoint overhead.
