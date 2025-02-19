import copy
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OriginalData:
    """
    Class to save data in its original form for later reference and reuse.
    This makes the original data immutable.
    """

    X_cat: dict = None
    X_cont: dict = None
    y: dict = None

    def get_data(self):
        return (
            copy.deepcopy(self.X_cat),
            copy.deepcopy(self.X_cont),
            copy.deepcopy(self.y),
        )

    def _get_split_data(self, split):
        return (
            copy.deepcopy(self.X_cat[split]),
            copy.deepcopy(self.X_cont[split]),
            copy.deepcopy(self.y[split]),
        )

    def get_train_data(self):
        return self._get_split_data("train")

    def get_val_data(self):
        return self._get_split_data("val")

    def get_test_data(self):
        return self._get_split_data("test")

    def get_total_obs(self):
        train_obs = self.get_train_obs()
        val_obs = self.get_val_obs()
        test_obs = self.get_test_obs()
        return train_obs + val_obs + test_obs

    def get_train_obs(self):
        return (
            self.X_cat["train"].shape[0]
            if self.X_cat["train"] is not None
            else self.X_cont["train"].shape[0]
        )

    def get_val_obs(self):
        if self.X_cat["val"] is not None or self.X_cont["val"] is not None:
            return (
                self.X_cat["val"].shape[0]
                if self.X_cat["val"] is not None
                else self.X_cont["val"].shape[0]
            )
        else:
            return 0

    def get_test_obs(self):
        return (
            self.X_cat["test"].shape[0]
            if self.X_cat["test"] is not None
            else self.X_cont["test"].shape[0]
        )


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Adapted from: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, X_cat, X_cont, y, batch_size=32, shuffle=False, drop_last=False):
        self.dataset_len = X_cat.shape[0] if X_cat is not None else X_cont.shape[0]
        assert all(
            t.shape[0] == self.dataset_len for t in (X_cat, X_cont, y) if t is not None
        )
        self.X_cat = X_cat
        self.X_cont = X_cont
        self.y = y

        self.batch_size = batch_size
        self.shuffle = shuffle

        if drop_last:
            self.dataset_len = (self.dataset_len // self.batch_size) * self.batch_size

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        if self.indices is not None:
            indices = self.indices[self.i : self.i + self.batch_size]
            batch = {}
            batch["X_cat"] = (
                torch.index_select(self.X_cat, 0, indices)
                if self.X_cat is not None
                else None
            )
            batch["X_cont"] = (
                torch.index_select(self.X_cont, 0, indices)
                if self.X_cont is not None
                else None
            )
            batch["y"] = (
                torch.index_select(self.y, 0, indices) if self.y is not None else None
            )

        else:
            batch = {}
            batch["X_cat"] = (
                self.X_cat[self.i : self.i + self.batch_size]
                if self.X_cat is not None
                else None
            )
            batch["X_cont"] = (
                self.X_cont[self.i : self.i + self.batch_size]
                if self.X_cont is not None
                else None
            )
            batch["y"] = (
                self.y[self.i : self.i + self.batch_size]
                if self.y is not None
                else None
            )

        self.i += self.batch_size

        batch = tuple(batch.values())
        return batch

    def __len__(self):
        return self.n_batches
