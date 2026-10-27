"""Save a wandb workspace view for the Sudoku runs: runs grouped by condition, mean ± std over seeds.

Needs `pip install wandb-workspaces`. Usage: python scripts/wandb_workspace_sudoku.py [--entity puma-mbz] [--project puma-sudoku]
  [--url <saved view url>]  (with --url the existing view is overwritten instead of saving a new one)
"""
import argparse

import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

DECODERS = [f"top_k{m}_unmasking_{k}" for m in ("", "_margin") for k in (1, 2, 3)]


def line(title, y, **kw):
    return wr.LinePlot(title=title, y=y, groupby_aggfunc="mean", groupby_rangefunc="stddev", **kw)


def bar(title, metric, **kw):
    # bar plots use each run's summary, i.e. the last logged value (the final step)
    return wr.BarPlot(title=title, metrics=[metric], orientation="h", groupby_aggfunc="mean", groupby_rangefunc="stddev", **kw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entity", default="puma-mbz")
    p.add_argument("--project", default="puma-sudoku")
    p.add_argument("--url", help="existing view to overwrite")
    p.add_argument("--name", default="Conditions grouped, mean and std over seeds")
    args = p.parse_args()

    acc_range = (0, 1)
    sections = [
        ws.Section(name="Accuracy (proposal decoder, top_k_unmasking_1)", is_open=True, panels=[
            line("accuracy: fp32 for B, C, D; bf16 as logged for the rest", ["eval_main/acc_top_k_unmasking_1"],
                 range_y=acc_range),
            line("fp32 eval: B, D every 1k updates, C every 10k", ["eval_fp32/acc_top_k_unmasking_1"], range_y=acc_range),
            line("bf16 eval as logged (no B, D: their retrain logs fp32)", ["eval_bf16/acc_top_k_unmasking_1"], range_y=acc_range),
        ]),
        ws.Section(name="Final results (mean and std over seeds)", is_open=True, panels=[
            bar("final accuracy: fp32 for B, C, D; bf16 as logged for the rest", "eval_main/acc_top_k_unmasking_1", range_x=acc_range),
            bar("final accuracy: bf16 as logged (no B, D)", "eval_bf16/acc_top_k_unmasking_1", range_x=acc_range),
            bar("final val loss", "eval/val_loss"),
        ]),
        ws.Section(name="Loss", is_open=True, panels=[
            line("train loss", ["train/loss"], log_y=True, smoothing_type="exponential", smoothing_factor=0.6),
            line("val loss", ["eval/val_loss"], log_y=True),
            line("learning rate", ["train/lr"]),
        ]),
        ws.Section(name="fp32 accuracy, all decoders (B, D; C has top_k_unmasking_1 only)", is_open=False, panels=[
            line(d, [f"eval_fp32/acc_{d}"], range_y=acc_range) for d in DECODERS
        ]),
        ws.Section(name="bf16 accuracy, all decoders", is_open=False, panels=[
            line(d, [f"eval_bf16/acc_{d}"], range_y=acc_range) for d in DECODERS
        ]),
    ]
    # bar charts follow the runs-table order: best final accuracy first
    # hidden (runs stay in the project): Erepro (E seed1000 rerun, a reproducibility check) and the first B/D runs
    # (20261019, bf16 eval, 5 seeds), replaced by the fp32-eval retrain sudoku_{B,D}_20261022 (10 seeds)
    hidden = [ws.Config("condition") != "Erepro", ws.Config("experiment") != "sudoku_B_20261019",
              ws.Config("experiment") != "sudoku_D_20261019"]
    runset = ws.RunsetSettings(groupby=[ws.Config("condition")], filters=hidden,
                               order=[ws.Ordering(ws.Summary("eval_main/acc_top_k_unmasking_1"), ascending=False)])
    if args.url:
        view = ws.Workspace.from_url(args.url)
        view.name, view.sections, view.runset_settings = args.name, sections, runset
    else:
        view = ws.Workspace(entity=args.entity, project=args.project, name=args.name, sections=sections,
                            runset_settings=runset)
    view.save()
    print(view.url)


if __name__ == "__main__":
    main()
