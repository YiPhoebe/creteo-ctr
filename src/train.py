import itertools
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.metrics import log_loss, roc_auc_score

from src.datamodule import CriteoIterable
from src.models.deepfm import DeepFM
from src.models.logistic import LogisticCTR
from src.models.dlrm import DLRMmini
from src.utils import load_config, set_seed, get_device


def build_model(name: str, d_dense: int, bucket_sizes: list[int], emb_dim: int):
    name = name.lower()
    if name == "logistic":
        return LogisticCTR(d_dense, len(bucket_sizes), emb_dim, bucket_sizes)
    if name == "deepfm":
        return DeepFM(d_dense, bucket_sizes, emb_dim)
    if name == "dlrm":
        return DLRMmini(d_dense, bucket_sizes, emb_dim)
    raise ValueError(f"Unknown model: {name}")


def main(config_path: str = "configs/criteo_stream.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    dataset_name = cfg.get("dataset", "criteo/CriteoClickLogs")
    train_split = cfg.get("splits", {}).get("train", "train")
    val_split = cfg.get("splits", {}).get("val", "validation")
    sample_size = int(cfg.get("sample_size", 100000))
    bucket_size_default = int(cfg.get("bucket_size_default", 1_000_003))
    batch_size = int(cfg.get("batch_size", 2048))
    epochs = int(cfg.get("epochs", 3))
    lr = float(cfg.get("learning_rate", 1e-3))
    emb_dim = int(cfg.get("emb_dim", 16))

    # streaming dataset for quick stats estimation
    ds_stream = load_dataset(dataset_name, split=train_split, streaming=True)
    sample = list(itertools.islice(ds_stream, sample_size))

    if len(sample) == 0:
        raise RuntimeError("Empty sample from streaming dataset. Check dataset/split.")

    d_dense = len(sample[0]["dense_features"])  # type: ignore[index]
    n_cats = len(sample[0]["cat_features"])  # type: ignore[index]
    dense_mean = [
        sum(ex["dense_features"][i] for ex in sample) / len(sample) for i in range(dense_features := d_dense)
    ]
    # start with std = 1.0 for simplicity; refine later if desired
    dense_std = [1.0] * d_dense
    bucket_sizes = [bucket_size_default] * n_cats

    # recreate streams after consuming sample
    train_stream = load_dataset(dataset_name, split=train_split, streaming=True)
    val_stream = load_dataset(dataset_name, split=val_split, streaming=True)

    train_ds = CriteoIterable(train_stream, bucket_sizes, dense_mean, dense_std)
    val_ds = CriteoIterable(val_stream, bucket_sizes, dense_mean, dense_std)

    device = get_device()
    model_name = cfg.get("model", "deepfm")
    model = build_model(model_name, d_dense, bucket_sizes, emb_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss()

    def run_epoch(loader: DataLoader, max_steps: int = 2000):
        model.train()
        losses = []
        for step, (dense, cats, y) in enumerate(loader):
            if step >= max_steps:
                break
            dense, cats, y = dense.to(device), cats.to(device), y.to(device).unsqueeze(1)
            p = model(dense, cats)
            loss = bce(p, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        return sum(losses) / max(1, len(losses))

    def evaluate(loader: DataLoader, max_steps: int = 200):
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for step, (dense, cats, y) in enumerate(loader):
                if step >= max_steps:
                    break
                p = model(dense.to(device), cats.to(device)).cpu().numpy().ravel().tolist()
                ys += y.numpy().tolist()
                ps += p
        return log_loss(ys, ps), roc_auc_score(ys, ps)

    train_loader = DataLoader(train_ds, batch_size=batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    for epoch in range(epochs):
        tr_loss = run_epoch(train_loader)
        va_logloss, va_auc = evaluate(val_loader)
        print(
            f"[Epoch {epoch}] train_loss={tr_loss:.5f}  val_logloss={va_logloss:.5f}  val_auc={va_auc:.5f}"
        )


if __name__ == "__main__":
    config = os.environ.get("CRITEO_CTR_CONFIG", "configs/criteo_stream.yaml")
    main(config)

