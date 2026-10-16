# Sudoku: options A and E

Two training conditions from the course proposal ("What Makes Progressive Masking Efficient?",
Table 1), built on the official PUMA code with minimal changes. Upstream base commit: `2f97f87`.

| Option | Config | What it is |
|---|---|---|
| A, random masking | `yaml_files/sudoku_A_random.yaml` | PUMA's `sudoku_baseline.yaml` (`strategy: standard`). Every update uses a fresh example. The mask count is uniform in 1..L_eff, positions are uniform, and the loss is divided by the number of masked cells. No code change was needed. |
| E, confidence progression | `yaml_files/sudoku_E_confidence.yaml` | PUMA's progressive pool with a fixed 4-state schedule. Each example is reused for 4 nested states with masked counts (81, 61, 41, 21)/81 of its empty cells. Each transition reveals the highest-confidence masked cells (max_v p) with their true digits. Ties are broken at random, there is no confidence thresholding, and batch slots start at staggered stages. |

L_eff is the number of empty cells. PUMA's single-sequence Sudoku keeps the 20–29 givens visible as
prompt positions, so L_eff is 52–61. The proposal's counts (81, 61, 41, 21) out of 81 are therefore
rescaled per puzzle, giving masked fractions of 1.000, 0.753, 0.506 and 0.259. For example, a puzzle
with L_eff = 57 is trained at 57, 43, 29 and 15 masked cells.

Stock PUMA (`sudoku_puma.yaml`) differs from E in two ways: it uses K = 10 (8 stochastic stages)
and confidence fast-forwarding at τ = 0.9. That is the proposal's separate "PUMA reference"
configuration, which has not been run yet.

## Code changes vs upstream PUMA
- `progressive.py`: optional `training.mask_schedule` (fixed per-stage mask counts) and
  `training.random_tie_break`. Both are off by default, so upstream behaviour is unchanged.
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

1. **Environment.** Any Python with a CUDA build of PyTorch 2.x works. We used a venv layered on an
   existing conda env (PyTorch 2.13 + cu129), so torch is not reinstalled:
   ```bash
   python -m venv --system-site-packages .venv
   .venv/bin/pip install omegaconf wandb datasets transformers gdown
   ```
2. **Data.** Download PUMA's Sudoku files into `data/sudoku_new/`, then build the caches.
   `data/sudoku.py` would otherwise convert ~1.8M puzzles serially on first use, which takes
   about 85 min, and each DDP rank would race to write the same file.
   ```bash
   .venv/bin/gdown --folder https://drive.google.com/drive/folders/1VNg6G9faB4FIoFv_9ktspYHURokLKumb -O data/sudoku_new
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
   Each run directory `{A,E}/seed<s>/` contains:
   - `run_info.txt`: Slurm job, node, GPU UUIDs, git state.
   - `code.diff`, the config copy and `config.yaml`.
   - `train.log` and `metrics.jsonl`.
   - Checkpoints every 10k steps plus the final step (`step=*.pt`, `ema_step=*.pt`).

The launcher runs a CUDA preflight (a real matmul on each allocated GPU) before training. It never
changes `CUDA_VISIBLE_DEVICES`. Its `--exclude` list holds nodes that failed CUDA startup or were
reported as bad; `scripts/cuda_diag.sbatch` is a read-only diagnostic for checking a node.

## Status (2026-10-16)
Seeds 1000–5000 for A and E were submitted as Slurm jobs 316850 and 316851. Outputs are written to
`runs/sudoku_AE_20261016/` on the cluster, and `runs/` is not tracked in git.

Open items:
- Summarise the runs: updates and training time to 50/75/90% exact-grid accuracy (two consecutive
  evals at or above the target), final accuracy, and paired A−E differences.
- Run the remaining proposal conditions (B, C, D, F) and the PUMA reference configuration.
- Proposal deviations to resolve: a validation split separate from the test set, removing duplicate
  puzzles across splits, and per-cell accuracy.
