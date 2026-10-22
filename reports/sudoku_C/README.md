# Sudoku condition C: five-seed results and W&B import

Condition C reuses each example across four scheduled states and independently
resamples the masked positions at every state. Previously visible target cells
can become masked again. Givens remain visible and revealed targets use the
ground truth. The shared mask-count schedule is scaled to the puzzle's empty
cells, following the team's implementation.

All five seeds completed **53,570 optimizer updates**. Training used BF16 mixed
precision with FP32 parameters and optimizer states. The primary results below
use **full FP32 evaluation of the final EMA checkpoint**, with the
`top_k_unmasking_1` decoder on the first 999 supplied test puzzles.

## Final results

| Seed | Full FP32 evaluation (%) | Original BF16 evaluation (%) |
|---|---:|---:|
| 1000 | 82.8829 | 36.2362 |
| 2000 | 76.6767 | 83.2833 |
| 3000 | 93.7938 | 90.7908 |
| 4000 | 91.6917 | 86.0861 |
| 5000 | 67.3674 | 51.1512 |
| **Mean ± sample SD** | **82.4825 ± 10.8983** | **69.5095 ± 24.2978** |

SD uses `ddof=1`; the units of SD in this table are percentage points. Results
use one fixed evaluation mode and the same final update for every seed.
BF16-forward/FP32-softmax evaluation is a separate final-checkpoint diagnostic:
82.6426% mean, 10.7694 percentage points sample SD.

Suggested row for the team's existing report table:

| Cond. | Final acc. (%) | BF16 eval as logged (%) | Final val loss |
|---|---:|---:|---:|
| C | 82.5 ± 10.9 (FP32) | 69.5 ± 24.3 | 0.076 ± 0.005 |

The recorded generation metric checks Sudoku validity and preservation of
givens; it does not compare every generated cell to the stored solution. The
generation set comes from the supplied test split. Validation loss uses a
separate held-out random-mask split and non-EMA weights.

## Files

- `runs.json`: five self-contained W&B run configurations, histories and summaries.
- `upload_C.py`: importer; default execution validates locally without uploading.
- `test_upload_C.py`: three existing checks of the import validation contract.
- `history.csv`: 2,680 merged training/evaluation rows, with seed and update.
- `final_per_seed.csv`: all three evaluation modes at the final checkpoint, in percent.
- `final_summary.json`: means and sample SDs for those three modes, in percent units.
- `precision_rescores.csv`: the 40 saved C precision evaluations with provenance.
- `all_conditions_std.csv`: A–F final summary with evaluation modes kept separate.
- `comparison_fp32/`: C/B/D final comparison and available checkpoint scores.
- `comparison_bf16/`: original A/C/F comparisons, histories and precision diagnostics.
- `provenance/`: the source patch and completed study manifest from the isolated run checkout.
- `validation.json`: checks performed when preparing the original import package.
- `FILES.sha256`: checksums of the committed package contents, excluding this checksum file.

The source patch is archived for reproducibility; it has not been applied to
this results branch. Training checkpoints and datasets remain in the original
experiment locations. The patch is based on
`fe967e2bbb613ed5a1aab903e7c5e111d159b0a6` (`sudoku-a-e-options`).

## Import into W&B

No W&B upload was performed while preparing this package. Importing creates
five separate runs in `puma-mbz/puma-sudoku`, with `condition=C` and group `C`.
It requires project write access, but no GPU or retraining.

From `/home/anton.egorov/PUMA`, use the existing experiment environment:

```bash
cd /home/anton.egorov/PUMA
/home/anton.egorov/PUMA-C/.venv/bin/python reports/sudoku_C/upload_C.py
/home/anton.egorov/PUMA-C/.venv/bin/wandb login
mkdir -p reports/sudoku_C/.wandb-runtime
WANDB_DIR="$PWD/reports/sudoku_C/.wandb-runtime" /home/anton.egorov/PUMA-C/.venv/bin/python reports/sudoku_C/upload_C.py --upload
```

On another machine, use Python 3.11 with `wandb==0.30.0`, authenticate with your
own account, and run the same script. Use the team's configured W&B endpoint.
The Forge report sharing link is not an API endpoint or an API key.

Run IDs are `C-seed1000-20261022` through `C-seed5000-20261022`. Existing IDs
cause an error (`resume="never"`), so running the importer twice will not
overwrite the first import. Completed uploads are recorded in the ignored
`upload_receipts.jsonl`. If an upload is interrupted, inspect the receipts and
project before importing only remaining seeds with `--seeds`. A partially
created failed run needs review before retrying that seed.

To rerun the local validation tests:

```bash
/home/anton.egorov/PUMA-C/.venv/bin/python -m unittest discover -s reports/sudoku_C -p test_upload_C.py -v
```

## Plotting in the shared report

Include all five C runs and group by `config.condition`. Use optimizer updates
(`optimizer_step` or `_step`) on the x-axis. The payload preserves original
update numbers and logs accuracy as fractions in [0, 1].

| Metric | Coverage per seed |
|---|---|
| `eval_main/acc_top_k_unmasking_1` | Six full-FP32 checkpoint observations |
| `eval_fp32/acc_top_k_unmasking_1` | The same six full-FP32 observations |
| `eval_bf16/acc_top_k_unmasking_1` | 54 original BF16 observations |
| `eval_bf16_fp32_softmax/acc_top_k_unmasking_1` | Final checkpoint only |
| `train/loss`, `train/lr`, `eval/val_loss` | Original logged observations |

Full-FP32 observations exist at updates 10,000, 20,000, 30,000, 40,000, 50,000
and 53,570. Missing observations are not filled with BF16 values or interpolated
data. These curves do not establish threshold-crossing times at 1,000-update
resolution. Other decoders have BF16 results only.

The main report groups seeds into a mean with a standard-deviation band; the
precision diagnostic panel also compares individual runs. Add C to any
explicit B/D-only diagnostic filters and update the chart captions accordingly.
The numeric table above uses sample SD; retain that definition when calculating
tables outside W&B. Editing the shared report requires report-edit access.

B/D use full-FP32 main evaluation in the existing report. A/E/F remain
BF16-evaluated, so preserve the precision labels when extending the comparison.
B/D retrospective evaluation hardware and batch-layout details are undocumented
in the available source. This package preserves both C evaluation modes rather
than selecting a different mode for each seed.

## Sources

- [Shared team report](https://forge.coreweave.com/wandb/puma-mbz/puma-sudoku/reports/Sudoku-progression-conditions-final-accuracy-and-the-bf16-eval-issue--VmlldzoxODE2MjE5Ng)
- [W&B quickstart](https://docs.coreweave.com/products/wandb/quickstart)
- [W&B report editing](https://docs.coreweave.com/products/wandb/reports/edit-a-report)
- [W&B Run API](https://docs.wandb.ai/ref/python/experiments/run/)

The original importer was validated with SDK 0.30.0; online upload was not
tested. See `validation.json` for the original validation record.
