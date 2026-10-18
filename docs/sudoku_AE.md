# Sudoku: options A, E, F and the PUMA reference

Training conditions from the course proposal ("What Makes Progressive Masking Efficient?",
Table 1), plus stock PUMA as a reference. All are built on the official PUMA code with minimal
changes. Upstream base commit: `2f97f87`.

| Option | Config | What it is |
|---|---|---|
| A, random masking | `yaml_files/sudoku_A_random.yaml` | PUMA's `sudoku_baseline.yaml` (`strategy: standard`). Every update uses a fresh example. The mask count is uniform in 1..L_eff, positions are uniform, and the loss is divided by the number of masked cells. No code change was needed. |
| E, confidence progression | `yaml_files/sudoku_E_confidence.yaml` | PUMA's progressive pool with a fixed 4-state schedule. Each example is reused for 4 nested states with masked counts (81, 61, 41, 21)/81 of its empty cells. Each transition reveals the highest-confidence masked cells (max_v p) with their true digits. Ties are broken at random, there is no confidence thresholding, and batch slots start at staggered stages. |
| F, Gt-Margin progression | `yaml_files/sudoku_F_gtmargin.yaml` | Identical to E except for the reveal rule. Cells are ranked by the ground-truth margin p(x0) − max_{v≠x0} p(v) (Asano et al.), computed from the training forward pass. Because the rule uses the hidden label, it avoids revealing cells the model is confidently wrong about. |
| P, PUMA reference | `yaml_files/sudoku_PUMA_reference.yaml` | Stock `sudoku_puma.yaml`: K = 10 (8 stages with randomly sampled reveal counts) and confidence fast-forwarding at τ = 0.9. Exact ties are broken by PUMA's default `topk` order. |

L_eff is the number of empty cells. PUMA's single-sequence Sudoku keeps the 20–29 givens visible as
prompt positions, so L_eff is 52–61. The proposal's counts (81, 61, 41, 21) out of 81 are therefore
rescaled per puzzle, giving masked fractions of 1.000, 0.753, 0.506 and 0.259. For example, a puzzle
with L_eff = 57 is trained at 57, 43, 29 and 15 masked cells.

P differs from E in its schedule (8 stochastic stages instead of 4 fixed ones) and in
fast-forwarding cells above τ = 0.9. B, C and D from the proposal are not implemented yet.

## Code changes vs upstream PUMA
- `progressive.py`: optional `training.mask_schedule` (fixed per-stage mask counts),
  `training.random_tie_break`, and `training.reveal_score` (`confidence` or `gt_margin`). The
  defaults (no schedule, no tie-breaking, `confidence`) keep upstream behaviour unchanged.
- `train.py`:
  - OmegaConf dotlist overrides on the command line (`training.seed=…`, `training.ckpt_dir=…`).
  - Per-run output directory with `config.yaml` and `metrics.jsonl`. `train_time_s` excludes evaluation time.
  - DistributedSampler seeded by the training seed. Upstream uses seed 0, so data order was identical across seeds.
  - Evaluation and checkpoint at the final step.

## Shared settings (PUMA paper Table 2 / sudoku YAMLs)
- Model: 8 layers, hidden 256, MLP 768, 8 heads, length 81.
- Optimiser: AdamW lr 3e-4, weight decay 0.01, cosine schedule with 1000 warmup steps, EMA 0.9999, grad-norm clip 1.
- Batch: 32 per GPU × 2 GPUs (DDP) = 64, bf16 autocast.
- Length: 2 epochs = 53,570 updates for both options.
- Data: PUMA's Sudoku files (1,804,462 train puzzles, 99,999 test puzzles). 5% of train is held out
  for validation loss, using split seed 2026.

Evaluation uses PUMA's built-in protocol, run every 1000 updates and at the final step. (The PUMA
config uses 2000; 1000 is the proposal's interval.) The model is evaluated with EMA weights on the
first 1% of the test set (999 puzzles). Exact-grid accuracy is reported for six greedy decoders:
{top_k, top_k_margin} × {1, 2, 3} cells per step. The proposal's decoder is `top_k_unmasking_1`.
Validation loss uses random masks on the held-out split with non-EMA weights.

On A100-40GB, training takes about 37 s per 1000 updates and each evaluation about 72–87 s,
so one run takes about 1.6–1.8 h.

## Reproducing / continuing
All commands run from the repo root on the cluster.

1. **Environment.** `scripts/setup_env.sh` creates `.venv` from `requirements.txt`. The versions
   are pinned to those used for the runs, and torch is the CUDA 12.9 build, which works with the
   cluster's driver 570. It needs Python >= 3.10, but the login node's default `python3` is 3.8,
   so pass one explicitly:
   ```bash
   conda create -y -n puma python=3.11
   BASE_PYTHON=$(conda run -n puma which python) bash scripts/setup_env.sh
   ```
   To reuse an interpreter that already has a CUDA torch, add `REUSE_TORCH=1`. This is how the
   runs' `.venv` was built, on a conda env with torch 2.13.0+cu129. Check CUDA on a compute node
   with `mkdir -p runs/logs && sbatch scripts/cuda_diag.sbatch`.
2. **Data.** `scripts/download_sudoku.sh` fetches PUMA's two Sudoku files from Google Drive into
   `data/sudoku_new/` (about 5 GB) and checks their shapes. Then build the training caches on a CPU
   node. `data/sudoku.py` would otherwise convert ~1.8M puzzles serially on first use, which takes
   about 85 min, and each DDP rank would race to write the same file.
   ```bash
   bash scripts/download_sudoku.sh
   mkdir -p runs/logs
   sbatch scripts/preprocess_sudoku.sbatch
   ```
   This writes `train_mdm.npy` and `test_mdm.npy` (about 4 min on 32 CPUs).
3. **Smoke test** (10 min per option on the debug QoS):
   ```bash
   EXP_DIR=runs/smoke RUN_TIMEOUT=600 sbatch --qos=gpu-debug-qos --time=00:30:00 scripts/run_sudoku_AE.sbatch A:1 E:1
   ```
4. **Runs.** Each job runs its `COND:SEED` items sequentially on 2 GPUs and skips items that
   already have a `DONE` file, so a failed job can be resubmitted with the same items:
   ```bash
   EXP_DIR=runs/sudoku_AE sbatch scripts/run_sudoku_AE.sbatch A:1000 E:1000 A:2000 E:2000 A:3000
   EXP_DIR=runs/sudoku_AE sbatch scripts/run_sudoku_AE.sbatch E:3000 A:4000 E:4000 A:5000 E:5000
   ```
   Conditions are `A`, `E`, `F` and `P`; the F/P runs used the same commands with `P`/`F` items.
   ```bash
   EXP_DIR=runs/sudoku_PF sbatch scripts/run_sudoku_AE.sbatch P:1000 F:1000 P:2000 F:2000 P:3000
   EXP_DIR=runs/sudoku_PF sbatch scripts/run_sudoku_AE.sbatch F:3000 P:4000 F:4000 P:5000 F:5000
   ```
   Each run directory `{A,E}/seed<s>/` contains:
   - `run_info.txt`: Slurm job, node, GPU UUIDs, git state.
   - `code.diff`, the config copy and `config.yaml`.
   - `train.log` and `metrics.jsonl`.
   - Checkpoints every 10k steps plus the final step (`step=*.pt`, `ema_step=*.pt`).

The launcher runs a CUDA preflight (a real matmul on each allocated GPU) before training. It never
changes `CUDA_VISIBLE_DEVICES`. Its `--exclude` list holds nodes that failed CUDA startup or were
reported as bad; `scripts/cuda_diag.sbatch` is a read-only diagnostic for checking a node.

## Status (2026-10-17)
**A and E (done).** All 10 runs finished: jobs 316850 and 316851, outputs in
`runs/sudoku_AE_20261016/`. The table uses the `top_k_unmasking_1` decoder on 999 test puzzles,
with mean ± std over 5 seeds. A target counts as reached at the first of two consecutive evals at
or above it.

| | Final exact-grid acc. | Updates to 50% | Updates to 75% | Updates to 90% |
|---|---|---|---|---|
| A | 88.1% ± 1.8 | 7.4k ± 0.5k | 16.0k ± 2.0k | reached by 2/5 seeds (~42k) |
| E | 95.4% ± 0.6 | 5.6k ± 0.5k | 12.2k ± 1.9k | 22.8k ± 1.6k |

E beat A on every paired seed (+5 to +10 points final accuracy). Both cost about 37 ms per update.

**P and F (done).** All 10 runs finished: jobs 321634 (P:1000 F:1000 P:2000 F:2000 P:3000) and
321635 (F:3000 P:4000 F:4000 P:5000 F:5000), outputs in `runs/sudoku_PF_20261017/`. P is stock PUMA
with collapse (confidence fast-forwarding at τ = 0.9); E and F run without it.

| | Final exact-grid acc. | Updates to 50% | Updates to 75% | Updates to 90% |
|---|---|---|---|---|
| P | 95.4% ± 1.2 | 7.6k ± 2.1k | 13.2k ± 3.4k | 23.4k ± 5.9k |
| F | 95.7% ± 1.3 | 5.2k ± 0.4k | 11.2k ± 1.6k | 22.6k ± 5.3k |

E, P and F finish within seed noise of each other, about 7–8 points above A on every seed.
F reaches 50% and 75% slightly sooner than E on every seed, with no consistent difference at 90%.

**Results for analysis.** `results/sudoku/` (tracked in git) has per-eval, per-step and per-run
CSVs, a summary table, and the raw per-run logs. Regenerate it with
`python scripts/export_sudoku_results.py`; checkpoints stay in `runs/` on the cluster.

Open items:
- Learning-curve plots and paired statistical tests (data in `results/sudoku/`).
- Implement and run B, C and D. If F beats E, also run the top-2 margin (label-free) control.
- Proposal deviations to resolve: a validation split separate from the test set, removing duplicate
  puzzles across splits, and per-cell accuracy.
