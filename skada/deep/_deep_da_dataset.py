# %%
import torch
from sklearn.utils.validation import check_array
from skorch.utils import to_tensor
from torch.utils.data import Dataset

# %%
_EMPTY_ = torch.Tensor()
_DEFAULT_SAMPLE_DOMAIN_ = 0
# I don't really like that the _NO_LABEL_ value is torch.nan,
# but it didn't work with None
_NO_LABEL_ = torch.nan


class DeepDADataset(Dataset):
    def __init__(
        self,
        dataset=None,
        y=None,
        sample_domain=None,
        sample_weights=None,
        device="cpu",
    ):
        self.X = _EMPTY_
        self.y = _EMPTY_
        self.sample_domain = _EMPTY_
        self.sample_weights = _EMPTY_
        self.has_y = _EMPTY_
        self.has_weights = bool(len(self.sample_weights))
        self.device = device

        if sample_domain is None:
            sample_domain = _DEFAULT_SAMPLE_DOMAIN_

        if dataset is not None:
            if isinstance(dataset, dict) and len(dataset):
                self._if_given_dict(dataset)
            elif isinstance(dataset, DeepDADataset):
                self.merge(dataset)
            else:
                try:
                    X = check_array(dataset, allow_nd=True, ensure_2d=False)
                    X = to_tensor(X, device)
                except TypeError:
                    raise TypeError(
                        "Invalid dataset type. Expected a dict of the form "
                        "{'X', 'y'(optional), 'sample_domain'(optional), "
                        "'sample_weights'(optional)},"
                        " another DeepDADataset instance or "
                        "a data type convertible to a torch Tensor"
                        f"but found {type(dataset)} instead."
                    )
                self._true_init(X, y, sample_domain, sample_weights)

    def _if_given_dict(self, dataset: dict):
        if "X" not in dataset or "sample_domain" not in dataset:
            raise ValueError(
                "dataset represented as dict should contain both 'X' and"
                " 'sample_domain' keys."
            )
        sample_domain = dataset["sample_domain"]
        X = dataset["X"]
        y = dataset.get("y")
        sample_weights = dataset.get("sample_weights")

        dataset = DeepDADataset(X, y, sample_domain, sample_weights, device=self.device)

        self.merge(dataset)

    def _true_init(self, X, y, sample_domain, sample_weights):
        if isinstance(sample_domain, int):
            sample_domain = torch.full_like(X, sample_domain)
        else:
            sample_domain = check_array(sample_domain, allow_nd=True, ensure_2d=False)
            sample_domain = to_tensor(sample_domain, self.device)

        if y is None:
            y = torch.full_like(X, _NO_LABEL_)
            has_y = torch.full_like(X, False)
        else:
            y = check_array(
                y, allow_nd=True, force_all_finite="allow-nan", ensure_2d=False
            )
            y = to_tensor(y, self.device)
            has_y = y != _NO_LABEL_

        if sample_weights is None:
            sample_weights = _EMPTY_
            has_weights = False
        else:
            sample_weights = check_array(sample_weights)
            sample_weights = to_tensor(sample_weights, device=self.device)
            has_weights = True

        self.X = X
        self.y = y
        self.sample_domain = sample_domain
        self.sample_weights = sample_weights
        self.has_y = has_y
        self.has_weights = has_weights

    def merge(self, other: "DeepDADataset", keep_weights=False):
        if isinstance(other, DeepDADataset):
            self.X = torch.cat((self.X, other.X))
            self.y = torch.cat((self.y, other.y))
            if keep_weights:
                # printing a warning that the concatenated weights should form a
                # proper distribution would be nice here
                self.sample_weights = torch.cat(
                    (self.sample_weights, other.sample_weights)
                )
            else:
                self.sample_weights = _EMPTY_
            self.sample_domain = torch.cat((self.sample_domain, other.sample_domain))
            self.has_y = torch.cat((self.has_y, other.has_y))
            self.has_weights = bool(len(self.sample_weights))

        else:
            raise TypeError("Can only merge two instances of DeepDADataset")

    def __add__(self, other: "DeepDADataset"):
        self.merge(other)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if not isinstance(index, (int, slice)):
            raise TypeError(
                "DeepDADataset indices must be integers," f" not {type(index)}"
            )

        X = {"X": self.X[index], "sample_domain": self.sample_domain[index]}

        if self.has_weights:
            X["sample_weight"] = self.sample_weights[index]

        return X, self.y[index]

    def __repr__(self):
        rep = f"DeepDADataset(\n{self.X},\n\n{self.y},\n\n{self.sample_domain}"
        if self.has_weights:
            rep += f",\n\n{self.sample_weights}"
        rep += "\n    )"
        return rep

    def as_dict(self):
        dataset = {"X": self.X, "y": self.y, "sample_domain": self.sample_domain}
        if self.has_weights:
            dataset["sample_weights"] = self.sample_weights
        return dataset

    def as_arrays(self, return_weights=False):
        if self.has_weights and return_weights:
            return self.X, self.y, self.sample_domain, self.sample_weights
        else:
            return self.X, self.y, self.sample_domain

    def select_source(self, return_weights=False):
        mask = self.sample_domain >= 0
        dataset = self.as_arrays(return_weights=return_weights)
        dataset = (data[mask] for data in dataset)
        return DeepDADataset(*dataset, device=self.device)

    def select_target(self, return_weights=False):
        mask = self.sample_domain < 0
        dataset = self.as_arrays(return_weights=return_weights)
        dataset = (data[mask] for data in dataset)
        return DeepDADataset(*dataset, device=self.device)

    def select_domain(self, domain_id, return_weights=False):
        mask = self.sample_domain == domain_id
        dataset = self.as_arrays(return_weights=return_weights)
        dataset = (data[mask] for data in dataset)
        return DeepDADataset(*dataset, device=self.device)

    def per_domain_split(self, return_weights=False):
        dataset = {}
        for domain_id in self.sample_domain.unique():
            dataset[domain_id] = self.select_domain(domain_id, return_weights)
        return dataset

    def select_labels(self, return_weights=False):
        mask = self.has_y
        dataset = self.as_arrays(return_weights=return_weights)
        dataset = (data[mask] for data in dataset)
        return DeepDADataset(*dataset, device=self.device)
