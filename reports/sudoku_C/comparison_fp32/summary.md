# C versus B/D: separate full-fp32 checkpoint comparison

EMA, `top_k_unmasking_1`, 999 puzzles, final update 53,570. All values below use full-fp32 forward and softmax.
Means and sample SD use available matched seeds; any result with fewer than five seeds is provisional.

| Condition | Final re-scores available | Accuracy (%) |
|---|---:|---:|
| B | 5/5 | 79.399 ± 17.560 |
| C | 5/5 | 82.482 ± 10.898 |
| D | 5/5 | 74.134 ± 17.816 |

| Contrast | Matched final seed pairs | C minus baseline (percentage points) |
|---|---:|---:|
| C-B | 5/5 | 3.083 ± 13.240 |
| C-D | 5/5 | 8.348 ± 17.647 |

Positive paired accuracy differences favor C. Exact B/D intermediate histories are absent from the downloaded artifacts, so comparisons are restricted to the final update.

| C checkpoint update | Re-scores available | Accuracy (%) |
|---|---:|---:|
| 10000 | 5/5 | 62.162 ± 10.391 |
| 20000 | 5/5 | 79.179 ± 10.316 |
| 30000 | 5/5 | 83.864 ± 10.042 |
| 40000 | 5/5 | 84.705 ± 9.221 |
| 50000 | 5/5 | 83.123 ± 10.642 |
| 53570 | 5/5 | 82.482 ± 10.898 |

C intermediate scores describe checkpoint coverage only. No 1000-update threshold crossing or unsaved observation is inferred.

Source: [shared team report](https://forge.coreweave.com/wandb/puma-mbz/puma-sudoku/reports/Sudoku-progression-conditions-final-accuracy-and-the-bf16-eval-issue--VmlldzoxODE2MjE5Ng). Exact per-seed values and run IDs are in `final_per_seed.csv`; all missing checkpoint slots remain explicit in `checkpoint_coverage.csv`.

B/D re-score device, batch/distributed layout, TF32 settings, and test-data hash are undocumented in the available source. Their training settings are preserved in the baseline JSON and do not establish retrospective re-score execution settings.
