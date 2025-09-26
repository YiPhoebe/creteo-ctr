# Criteo CTR (1TB) — Streaming Pipeline Skeleton

This repo provides a minimal, ready-to-run scaffold for training CTR models on the Criteo 1TB click logs using Hugging Face Datasets streaming. It supports single-machine training and an optional DDP path.

## Structure

criteo-ctr/
- env.yml — Conda env
- configs/criteo_stream.yaml — Basic config
- src/
  - download_hf.py — HF streaming loader
  - preprocess.py — Hash bucket & normalization helpers
  - datamodule.py — PyTorch IterableDataset
  - models/
    - logistic.py — Logistic CTR baseline
    - deepfm.py — Simple DeepFM
    - dlrm.py — Mini DLRM (no interaction)
  - train.py — Single-GPU/CPU trainer
  - train_ddp.py — torchrun DDP trainer
  - evaluate.py — LogLoss/AUC evaluation
  - utils.py — Misc utilities

## Environment

Create and activate the environment:

```
conda env create -f env.yml
conda activate criteo_ctr
```

## Quick Start (Single Machine)

```
python -m src.train  # uses configs/criteo_stream.yaml by default
```

Environment override for config:

```
CRITEO_CTR_CONFIG=configs/criteo_stream.yaml python -m src.train
```

## DDP (Optional)

```
torchrun --nproc_per_node=NUM src/train_ddp.py
```

## Notes

- Data: Hugging Face `criteo/CriteoClickLogs` (24-day click logs). Streaming keeps memory usage small.
- Models: Logistic, DeepFM, and a mini DLRM. Replace with full DLRM and feature interactions for better performance.
- Preprocessing: Simple mean (sampled) and std=1.0 to start. For production, compute accurate stats (offline) and persist.
- Evaluation: Computes LogLoss and AUC on a streamed validation split. For DDP, the example averages loss across workers; extend as needed.

# creteo-ctr
# creteo-ctr
