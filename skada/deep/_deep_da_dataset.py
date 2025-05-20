# %%
import numpy as np
import torch
from skorch.utils import to_tensor
from torch.utils.data import Dataset

from skada import per_domain_split

# %%
_EMPTY_DOMAIN = [None]
_DEFAULT_DOMAIN_ID = 0


class DeepDADataset(Dataset):
    def __init__(self, *datasets, domain_id=None, **kwargs):
        self.X = torch.Tensor()
        self._domain_ids_container = [_DEFAULT_DOMAIN_ID]
        self._domain_sizes_container = [0]
        self.has_y = [False]
        self.has_weights = False
        self.y = _EMPTY_DOMAIN
        self.weights = torch.Tensor()

        assert domain_id is None or isinstance(domain_id, int), (
            "Keyword argument " "'domain_id' must be None or an integer."
        )
        if domain_id is None:
            domain_id = [_DEFAULT_DOMAIN_ID]

        if len(datasets) == 1:
            dataset = datasets[0]
            if dataset is not None and not isinstance(
                dataset, (dict, list, tuple, DeepDADataset, np.ndarray, torch.Tensor)
            ):
                raise TypeError(
                    "Invalid dataset type. Should be a dict of the form "
                    "{'X', 'y'(optional), 'domain'(optional), "
                    "'sample_weights'(optional)}, "
                    "a list or tuple of form "
                    "(X, y(optional), domain_id(optional), weights(optional)),"
                    " a numpy array, torch tensor, or another DeepDADataset instance."
                )
            elif isinstance(dataset, dict) and len(dataset):
                self._if_given_dict(dataset, **kwargs)
            elif isinstance(dataset, (list, tuple)) and len(dataset):
                self._if_given_list(dataset, domain_id, **kwargs)
            elif isinstance(dataset, DeepDADataset) and len(dataset):
                self.merge(dataset)
            elif isinstance(dataset, np.ndarray) and len(dataset):
                dataset = to_tensor(dataset, **kwargs)
                self._if_given_tensor(dataset, domain_id)
            elif isinstance(dataset, torch.Tensor) and len(dataset):
                self._if_given_tensor(dataset, domain_id)

        else:
            for dataset in datasets:
                dataset = DeepDADataset(dataset, domain_id=domain_id, **kwargs)
                self.merge(dataset)

    def _if_given_dict(self, dataset, **kwargs):
        if "X" not in dataset or "sample_domain" not in dataset:
            raise ValueError(
                "dataset represented as dict should contain both 'X' and"
                " 'sample_domain' keys."
            )
        sample_domain = dataset["sample_domain"]
        X = dataset["X"]
        x_dict, xt_dict = per_domain_split(X, sample_domain=sample_domain)
        x_dict.update(xt_dict)
        try:
            y = dataset["y"]
            ys_dict, yt_dict = per_domain_split(y, sample_domain=sample_domain)
            y_dict = ys_dict.update(yt_dict)
        except KeyError:
            y_dict = {k: _EMPTY_DOMAIN for k in x_dict.keys()}

        try:
            weights = dataset["sample_weights"]
            ws_dict, wt_dict = per_domain_split(weights, sample_domain=sample_domain)
            w_dict = ws_dict.update(wt_dict)
        except KeyError:
            w_dict = {k: _EMPTY_DOMAIN for k in x_dict.keys()}

        per_domain_dict = {k: [x_dict, y_dict, k, w_dict] for k in x_dict.keys()}
        for v in per_domain_dict.values():
            dataset = DeepDADataset(v, **kwargs)
            self.merge(dataset)

    def _if_given_list(self, dataset, domain_id, **kwargs):
        dataset = list(dataset)
        X = dataset[0]
        if isinstance(X, np.ndarray):
            X = to_tensor(X, **kwargs)
        elif not isinstance(X, torch.Tensor):
            raise TypeError(
                "Given data should be represented as a torch tensor or a numpy array "
                "(which will be converted to a tensor)."
            )
        dataset += [_EMPTY_DOMAIN, _EMPTY_DOMAIN, _EMPTY_DOMAIN]
        y = dataset[1]
        pre_domain_id = dataset[2]
        weights = dataset[3]
        self.has_y = y is not _EMPTY_DOMAIN
        self.has_weights = weights is not _EMPTY_DOMAIN and len(weights)
        self._domain_ids_container = (
            [domain_id] if pre_domain_id is _EMPTY_DOMAIN else [pre_domain_id]
        )
        self.y = y
        self.weights = weights
        self._domain_sizes_container = [len(self.X)]

    def _if_given_tensor(self, dataset, domain_id):
        self.X = dataset
        self._domain_ids_container = [domain_id]
        self._domain_sizes_container = [len(self.X)]

    def merge(self, other: "DeepDADataset", keep_weights=False):
        if isinstance(other, DeepDADataset):
            self.X = torch.cat(self.X, other.X)
            self._domain_ids_container += other.domain_id
            self._domain_sizes_container += other._domain_sizes_container
            self.y += other.y
            self.has_y += other.has_y
            if keep_weights:
                # printing a warning that the concatenated weights should be a
                # proper distribution would be nice here
                self.weights = torch.cat(self.weights, other.weights)
            self.has_weights = bool(len(self.weights))
        else:
            raise TypeError("Can only merge two instances of DeepDADataset")

    def __add__(self, other: "DeepDADataset"):
        self.merge(other)

    def __len__(self):
        return sum(
            self._domain_sizes_container,
        )

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError(f"Dataset indices must be integers, not {type(index)}")
        elif index >= len(self):
            raise IndexError("Dataset index out of range")
        x = self.X[index]
        _ = 0
        i = -1
        while _ < index:
            i += 1
            _ += self._domain_sizes_container[i]
        try:
            y_ind = index - sum(self._domain_sizes_container[:i])
            y_res = self.y[i][y_ind]
            return x, y_res
        except TypeError:
            return x, None


# TODO: make method for add_data, source_idx, target_idx,
# source_data, target_data, source_label, target_label,
# domain_separation, __repr__, as_dict, as_numpy, as_tensor,
#
# Would it work better with containers all the way up?
# The way I am seeing this is with the _DDAD_Container class being a series of
# containers that keeps the order of the data and the creation methods,
# and the DeepDADataset being a subclass working towards a better visualization
