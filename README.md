<div align="center">

<img src="figure/puma_title.svg" alt="PUMA: Progressive Unmasking for Accelerated Masked Diffusion Training" width="980" />

</div>

<p align="center">
  <a href="https://arxiv.org/abs/2602.10314">
    <img
      src="https://img.shields.io/badge/Paper-Arxiv-red?logo=arxiv&logoColor=red"
      alt="PUMA Paper on arXiv"
    />
  </a>
</p>

<p align="center">
  <img src="figure/fig_tinygsm_with_arm.svg" alt="PUMA Main Figure" width="700">
</p>

---

## Overview

**PUMA (Progressive Unmasking)** is a simple modification to the forward process of **Masked Diffusion Models (MDMs)**.
Instead of training on randomly masked sequences, PUMA aligns that training- and inference- time masking patterns, thereby focusingon inference-aligned masks and speeding up training. 

---
## Quick Start

### 1. Install Environment

```bash
# Create and activate conda environment
conda env create -f environment.yml
```

The slurm training script (see below) activates the environment automatically. To activate manually, run:

```bash
conda activate puma
```

### 2. Data preparation
We provide PUMA codebases for Sudoku and TinyGSM. 

- Sudoku: Download *sudoku-train-data.npy* and *sudoku-test-data.npy* from [here](https://drive.google.com/drive/folders/1VNg6G9faB4FIoFv_9ktspYHURokLKumb?usp=sharing) and put them in the `data/sudoku_new` folder. 

- TinyGSM: Run `data/tiny_gsm.py` that gives you files `labels.bin`, `meta.json`, and `prompt_mask.bin` for pretraining in the desired `out_dir` directory.

### 3. Run Training
Submit a job using the SLURM script:

```bash
sbatch job.sh
```

The SLURM script calls `train.py`, which handles the training loop.

### 4. Configuration

Config files are located in `yaml_files/`. Edit these YAML files to adjust:
- Model architecture
- Training hyperparameters  
- Dataset settings
- Logging options

We provide one config each for PUMA and the baseline for the following three settings: Sudoku, TinyGSM (standard), TinyGSM (block diffusion).

Please note that you need to update `wandb.entity` in the config, as well user-specific settings such as `account` and `partition` in `job.sh`, before launching a training run.

To train a block diffusion model, run `train_block.py` instead of `train.py`.

For ARM initialization, first pretrain an ARM model by setting `model.causal=true` in the config, and then initialize an MDM by setting `model.arm_init` in the config, and resetting `model.causal` to `false`.

### 5. Monitoring

Training logs and checkpoints are saved according to the paths specified in your config file. The training file also logs results to wandb.

---

## Explanation of each files
- `train.py`: unified file that handles the MDM pretraining (includes the vanilla MDM pretraining). Includes evaluation accuracy logging. This file is used for both Sudoku and TinyGSM, but not block diffusion.
- `train_block.py`: Training file for MDM pretraining in the block diffusion setting.
- `sampling.py`: sampling for a given MDM
- `progressive.py`': PUMA via batch streaming implementation. The implementation detail can be found in Section 3.2 and Appendix B.1.
- `progressive_block.py`: PUMA implementation for block diffusion. 
- `model`: our Qwen2-style attention implementations
- `eval`: eval util functions for Sudoku / TinyGSM

## Sudoku progression conditions
This fork adds Sudoku training conditions that separate the parts of PUMA. All of them use PUMA's Sudoku model and hyperparameters, and each one is a config in `yaml_files/`.

| Code | Config | Training grids |
|---|---|---|
| A | `sudoku_A_random.yaml` | Random masking (standard MDM training) |
| B | `sudoku_B_schedule.yaml` | Fixed 4-stage mask schedule, new puzzle every update |
| D | `sudoku_D_random.yaml` | Same puzzle for 4 stages, revealed cells stay revealed, random reveal order |
| E | `sudoku_E_confidence.yaml` | As D, but the most confident cells are revealed first |
| F | `sudoku_F_gtmargin.yaml` | As D, but cells are ranked by the ground-truth margin |
| P | `sudoku_PUMA_reference.yaml` | PUMA as published |
| PN | `sudoku_PUMA_nocollapse.yaml` | PUMA without confidence fast-forwarding |

The `*5_floor1.yaml` configs add a fifth stage with a single hidden cell.

Setup and training on Slurm, from the repo root:
```bash
BASE_PYTHON=/path/to/python3.11 bash scripts/setup_env.sh
bash scripts/download_sudoku.sh
mkdir -p runs/logs && sbatch scripts/preprocess_sudoku.sbatch
EXP_DIR=runs/sudoku sbatch scripts/run_sudoku_AE.sbatch A:1000 E:1000 F:1000
```
`run_sudoku_AE.sbatch` takes `COND:SEED` items for A, E, F, P and PN. B and D have their own launchers (`scripts/run_sudoku_B.sbatch`, `scripts/run_sudoku_D.sbatch`). `scripts/reeval_sudoku.py` re-scores saved checkpoints with an fp32 decode, and `scripts/export_sudoku_results.py` collects all runs into the CSV files in `results/sudoku/`.

## Citation
If you find this repository helpful, please consider citing our paper:

```bibtex
@misc{
    kim2026stoptrainingworstprogressive,
    title={{S}top {T}raining for the {W}orst: {P}rogressive {U}nmasking {A}ccelerates {M}asked {D}iffusion {T}raining},
    author={Jaeyeon Kim and Jonathan Geuter and David Alvarez-Melis and Sham Kakade and Sitan Chen},
    year={2026},
    eprint={2602.10314},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2602.10314}, 
}
```
