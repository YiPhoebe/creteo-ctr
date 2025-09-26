from typing import Iterable, List
from torch.utils.data import IterableDataset
import torch


class CriteoIterable(IterableDataset):
    def __init__(
        self,
        hf_stream: Iterable,
        bucket_sizes: List[int],
        dense_mean: List[float],
        dense_std: List[float],
    ):
        super().__init__()
        self.ds = hf_stream
        self.bucket_sizes = bucket_sizes
        self.dense_mean = dense_mean
        self.dense_std = dense_std

    def __iter__(self):
        for ex in self.ds:
            y = torch.tensor(ex["label"], dtype=torch.float32)
            dense = torch.tensor(
                [
                    (x - m) / (s if s > 0 else 1.0)
                    for x, (m, s) in zip(
                        ex["dense_features"], zip(self.dense_mean, self.dense_std)
                    )
                ],
                dtype=torch.float32,
            )
            cats = [hash(x) % b for x, b in zip(ex["cat_features"], self.bucket_sizes)]
            cats = torch.tensor(cats, dtype=torch.long)
            yield dense, cats, y

