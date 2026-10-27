"""Save a wandb report for the Sudoku runs: final accuracy, train/val loss, and the bf16 eval issue.

Built from the saved view made by scripts/wandb_workspace_sudoku.py. Needs `pip install wandb-workspaces`.
Numbers in the tables are from the run summaries (final step, 53,570 updates) and results/sudoku/conditions_BD.md;
C from reports/sudoku_C/README.md.
Usage: python scripts/wandb_report_sudoku.py [--entity puma-mbz] [--project puma-sudoku] [--url <report url>]
  (with --url the existing report is overwritten instead of saving a new one)
"""
import argparse

import wandb_workspaces.reports.v2 as wr

ACC = "acc_top_k_unmasking_1"
ACC_RANGE = (0, 1)

INTRO = """\
Sudoku, 8-layer MDM, 2 epochs (53,570 updates), batch 64, 5 seeds (1000..5000) per condition, 10 (1000..10000) for B
and D. Runs are grouped by
condition; plots show the mean over seeds with a ±1 std band. Accuracy is exact-grid accuracy of the proposal decoder
`top_k_unmasking_1` on the first 999 test puzzles, with EMA weights.

| Code | Condition |
|---|---|
| A | random masking |
| B | shared schedule, fresh examples (schedule-only control) |
| C | reused examples, independent masks |
| D | reused, nested, random reveal |
| E | confidence progression |
| F | Gt-Margin progression |
| P | PUMA reference |
| PN | PUMA without collapse |

`eval_main/*` is fp32 for B, C and D, otherwise the bf16 eval as logged during training. B and D were retrained with
the fixed eval (fp32 ranking every 1k updates); C was re-scored in fp32 at its saved checkpoints (every 10k updates).
See the bf16 section below for why the two differ."""

FINAL = """\
| Cond. | Final acc. (%) | bf16 eval as logged (%) | Final val loss |
|---|---|---|---|
| F | **95.7 ± 1.3** | 95.7 ± 1.3 | 0.078 ± 0.008 |
| E | **95.4 ± 0.6** | 95.4 ± 0.6 | 0.074 ± 0.005 |
| P | **95.4 ± 1.2** | 95.4 ± 1.2 | 0.076 ± 0.014 |
| PN | 93.4 ± 1.6 | 93.4 ± 1.6 | 0.089 ± 0.011 |
| A | 88.1 ± 1.8 | 88.1 ± 1.8 | 0.081 ± 0.008 |
| C | 82.5 ± 10.9 (fp32) | 69.5 ± 24.3 | 0.076 ± 0.005 |
| B | 81.7 ± 13.2 (fp32, 10 seeds) | n/a (fp32 in training) | 0.079 ± 0.008 |
| D | 79.1 ± 13.5 (fp32, 10 seeds) | n/a (fp32 in training) | 0.084 ± 0.010 |

Mean ± std over 5 seeds (10 for B and D) at the final step. B and D are the retrain with the fixed eval; seeds
1000..5000 reproduce the first runs' training exactly (first runs, fp32 re-score: B 79.4 ± 17.6, D 74.1 ± 17.8; bf16
as logged: B 62.2 ± 37.2, D 53.9 ± 37.6). A/E/F/P/PN are bf16-eval numbers (their checkpoints were not re-scored);
for those models the tie order barely matters (E seed 1000 in the ablation below: 94.3 bf16 vs 95.3 fp32)."""

LOSS = """\
Val loss is the random-mask MDM loss on a 5% held-out split, with non-EMA weights, so it is comparable across
conditions. Train loss is not: each condition trains on its own masking distribution (A on uniformly random masks,
the progression conditions on their mask schedules), so its level reflects how hard the training masks are, not
model quality."""

BF16 = """\
**Accuracies logged during training before the fix (A/C/E/F/P/PN and the first B/D runs) use a bf16 eval that is
biased against B, C and D.**

`sampling.mdm_sampling` ranks masked cells by `F.softmax(logits)` computed on bf16 logits, so the confidences are
bf16. Values ≥ ~0.998 all round to exactly 1.0 (bf16 spacing just below 1 is 2^-8), so in about 90% of decode
steps several cells tie for the top score, and `torch.topk` returns the lowest tied index (on CPU and CUDA).
Decoding then fills cells top-left first among near-equal confidences instead of by confidence.

E/F/P models are insensitive to this order. B, C and D models are not: they fail at the last few cells, and the bf16
eval reports far lower and much noisier accuracy than an fp32 decode of the same checkpoint (B seed 2000:
5.3% bf16 vs 51.1% fp32; D seed 1000: 35.0% vs 82.1%; C seed 1000: 36.2% vs 82.9%). The plots below use the runs
that have both: the first B/D runs (5 seeds each) and C."""

ABLATION = """\
**Ablation** (final checkpoints, CPU, 300 puzzles, exact-grid %): only the tie order matters.

| ranking | E seed 1000 | D seed 1000 | B seed 2000 |
|---|---|---|---|
| fp32 | 95.3 | 83.7 | 50.0 |
| bf16-rounded, lowest index on ties (= current eval) | 94.3 | **37.3** | **4.7** |
| bf16-rounded, highest index on ties | 94.3 | 89.3 | 70.7 |
| bf16-rounded, random tie-break | 95.7 | 86.0 | 60.7 |
| bf16 forward, fp32 softmax | 95.3 | 85.0 | 48.3 |

**Fix:** `p = F.softmax(logits.float(), dim=-1)` in `sampling.mdm_sampling` (keeps the bf16 forward). Checked on
GPU through the repo eval: on all 11 final checkpoints it is within 1.2 points of a full fp32 decode. Applied in
`ebad40a` (branch sudoku-b-d); the B/D retrain uses it.

**Caveats:** fp32 re-scores exist only at the saved checkpoints (every 10k updates), so the re-score curves (first
B/D runs, C) have 6 points; the retrained B/D runs have fp32 every 1k updates. A/E/F/P/PN checkpoints were not
re-scored. Some B/D seeds still decline late in fp32 (first runs: B 2000: 64.4 at 30k → 51.1; D 2000: 55.3 → 45.8;
D 4000: 76.3 → 69.8)."""


def line(title, y, **kw):
    return wr.LinePlot(title=title, y=y, groupby_aggfunc="mean", groupby_rangefunc="stddev", **kw)


def bar(title, metrics, **kw):
    # bar plots use each run's summary, i.e. the last logged value (the final step)
    return wr.BarPlot(title=title, metrics=metrics, orientation="h", groupby_aggfunc="mean",
                      groupby_rangefunc="stddev", **kw)


def runset(entity, project, name, filters, grouped=True):
    return wr.Runset(entity=entity, project=project, name=name, filters=filters,
                     groupby=["config.condition"] if grouped else [],
                     order=[wr.OrderBy(wr.SummaryMetric(f"eval_main/{ACC}"), ascending=False)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entity", default="puma-mbz")
    p.add_argument("--project", default="puma-sudoku")
    p.add_argument("--url", help="existing report to overwrite")
    p.add_argument("--title", default="Sudoku: progression conditions, final accuracy and the bf16 eval issue")
    args = p.parse_args()
    e, pr = args.entity, args.project

    # left out: Erepro (E seed1000 rerun, a reproducibility check) and, in the main plots, the first B/D runs
    # (20261019, bf16 eval), replaced by the fp32-eval retrain sudoku_{B,D}_20261022
    first_bd = "['sudoku_B_20261019', 'sudoku_D_20261019']"
    all_conds = runset(e, pr, "conditions", f"Config('condition') != 'Erepro' and Config('experiment') not in {first_bd}")
    # runs with both a bf16 eval and an fp32 re-score of the same checkpoints
    both = "Config('experiment') in ['sudoku_B_20261019', 'sudoku_D_20261019', 'sudoku_C_20261022']"
    rescored = runset(e, pr, "fp32 re-scored (first B/D runs, C)", both)
    rescored_runs = runset(e, pr, "fp32 re-scored runs (first B/D runs, C)", both, grouped=False)

    blocks = [
        wr.TableOfContents(),
        wr.H1("Setup"),
        wr.MarkdownBlock(INTRO),
        wr.H1("Final accuracy"),
        wr.MarkdownBlock(FINAL),
        wr.PanelGrid(runsets=[all_conds], panels=[
            bar("final accuracy (fp32 for B, C, D; bf16 otherwise)", [f"eval_main/{ACC}"], range_x=ACC_RANGE,
                layout=wr.Layout(x=0, y=0, w=24, h=10)),
            line("accuracy over training (fp32 for B, C, D; bf16 otherwise)", [f"eval_main/{ACC}"], range_y=ACC_RANGE,
                 layout=wr.Layout(x=0, y=10, w=24, h=10)),
        ]),
        wr.H1("Train and validation loss"),
        wr.MarkdownBlock(LOSS),
        wr.PanelGrid(runsets=[all_conds], panels=[
            line("train loss", ["train/loss"], log_y=True, smoothing_type="exponential", smoothing_factor=0.6,
                 layout=wr.Layout(x=0, y=0, w=24, h=10)),
            line("val loss (random masks, held-out split)", ["eval/val_loss"], log_y=True,
                 layout=wr.Layout(x=0, y=10, w=24, h=10)),
            bar("final val loss", ["eval/val_loss"], layout=wr.Layout(x=0, y=20, w=24, h=8)),
            line("learning rate", ["train/lr"], layout=wr.Layout(x=0, y=28, w=24, h=8)),
        ]),
        wr.H1("The bf16 eval problem"),
        wr.MarkdownBlock(BF16),
        wr.PanelGrid(runsets=[rescored_runs], panels=[
            bar("final accuracy per run: bf16 eval as logged vs fp32 re-score",
                [f"eval_bf16/{ACC}", f"eval_fp32/{ACC}"], range_x=ACC_RANGE,
                layout=wr.Layout(x=0, y=0, w=24, h=12)),
        ]),
        wr.PanelGrid(runsets=[rescored], panels=[
            line("first B/D runs and C: bf16 eval as logged", [f"eval_bf16/{ACC}"], range_y=ACC_RANGE,
                 layout=wr.Layout(x=0, y=0, w=24, h=10)),
            line("first B/D runs and C: fp32 re-score of the same checkpoints", [f"eval_fp32/{ACC}"], range_y=ACC_RANGE,
                 layout=wr.Layout(x=0, y=10, w=24, h=10)),
            bar("final accuracy (first B/D runs, C), bf16 as logged", [f"eval_bf16/{ACC}"], range_x=ACC_RANGE,
                layout=wr.Layout(x=0, y=20, w=24, h=8)),
            bar("final accuracy (first B/D runs, C), fp32 re-score", [f"eval_fp32/{ACC}"], range_x=ACC_RANGE,
                layout=wr.Layout(x=0, y=28, w=24, h=8)),
        ]),
        wr.MarkdownBlock(ABLATION),
    ]
    description = ("Final accuracy and losses per condition (mean ± std over 5 seeds, 10 for B and D), and why the "
                   "bf16 eval under-reports B, C and D.")
    if args.url:
        report = wr.Report.from_url(args.url)
        report.title, report.description, report.blocks = args.title, description, blocks
    else:
        report = wr.Report(entity=e, project=pr, title=args.title, width="fluid", blocks=blocks,
                           description=description)
    report.save()
    print(report.url)


if __name__ == "__main__":
    main()
