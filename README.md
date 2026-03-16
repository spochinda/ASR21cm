# ASR21cm — Arbitrary Super-Resolution for 21cm Cosmology

A deep learning framework for super-resolving 3D 21cm brightness temperature fields, built on top of [BasicSR](https://github.com/xinntao/BasicSR).

---

## Overview

This repository implements several approaches for super-resolving simulated 21cm cosmological fields:

| Model | Type | Description |
|---|---|---|
| `ArSSR` | Coordinate-based SR | Encoder (RDN/UNet) + implicit MLP decoder for arbitrary-scale SR |
| `ASR21cmModel` | Supervised SR | Standard super-resolution with L1/perceptual losses |
| `EsrASRGANModel` | GAN | ESRGAN-style adversarial training |
| `ScoreDiffusionVPSDEModel` | Diffusion | Score-based diffusion with VP-SDE |
| `ScoreDiffusionVPSDEvpredModel` | Diffusion | VP-SDE with v-prediction parameterisation |

---

## Repository Structure

```
ASR21cm/
├── archs/             # Network architectures (ArSSR, SongUNet, RDN, UNet)
├── data/              # Datasets (Custom21cmDataset, FixedScale21cmDataset)
├── losses/            # Custom losses (VPSDE, DSQ)
├── metrics/           # Evaluation metrics
├── models/            # Training logic for each model type
├── train.py           # Entry point (standard BasicSR pipeline)
└── train_asr.py       # Custom training pipeline for ArSSR

analysis/              # Post-training analysis scripts
options/               # YAML configuration files
datasets/              # Data directory (not tracked by git)
experiments/           # Training outputs and checkpoints
```

---

## Installation

```bash
git clone <repo-url>
cd ASR21cm
pip install -r requirements.txt
python setup.py develop
```

**Requirements:** Python 3.8+, PyTorch, basicsr, numpy, opencv-python.

---

## Data

The models expect 21cm brightness temperature cubes and initial condition (IC) cubes organised as:

```
datasets/
└── varying_IC/
    ├── T21_cubes/     # 21cm brightness temperature fields (Npix=256)
    └── IC_cubes/      # Initial condition density/velocity fields
```

Data is indexed by redshift and IC seed. See the dataset classes in [ASR21cm/data/](ASR21cm/data/) for the expected file format.

---

## Training

### Arbitrary Super-Resolution (ArSSR)

```bash
python ASR21cm/train_asr.py -opt options/ASR21cm_options_UNet.yml
```

### Fixed-Scale Diffusion Model (VP-SDE)

```bash
python ASR21cm/train.py -opt options/ScoreDiffusionvpred_options.yml
```

Resume an interrupted run:

```bash
python ASR21cm/train.py -opt options/ScoreDiffusionvpred_options.yml --auto_resume
```

### On a SLURM Cluster (Cambridge HPC)

Single GPU:
```bash
sbatch slurm_single_gpu
```

Multi-GPU (4x Ampere):
```bash
sbatch slurm_gpu
```

---

## Configuration

All options are controlled by YAML files in [options/](options/). Key settings:

| Option | Description |
|---|---|
| `model_type` | Which model class to use |
| `network_g.type` | Network architecture |
| `datasets.train.scale` | Upscaling factor |
| `datasets.train.redshifts` | List of redshifts to train on |
| `datasets.train.IC_seeds` | List of IC seeds for training |
| `beta_min` / `beta_max` | VP-SDE noise schedule bounds |
| `train.total_iter` | Total training iterations |

---

## Analysis

Post-training analysis scripts are in [analysis/](analysis/):

- `arbitrary_scaling.py` — evaluate ArSSR across a range of scales
- `redshift_scaling.py` — evaluate performance across redshifts
- `table_scaling.py` — generate summary tables of metrics

---

## Acknowledgements

Built on [BasicSR](https://github.com/xinntao/BasicSR). The `SongUNet` architecture is adapted from [NVlabs/edm](https://github.com/NVlabs/edm).
