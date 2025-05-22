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
        sample_weight=None,
        device="cpu",
    ):
        self.X = _EMPTY_
        self.y = _EMPTY_
        self.sample_domain = _EMPTY_
        self.sample_weight = _EMPTY_
        self.has_y = _EMPTY_
        self.has_weights = bool(len(self.sample_weight))
        self.device = device

        if sample_domain is None:
            sample_domain = _DEFAULT_SAMPLE_DOMAIN_

        if dataset is not None:
            if isinstance(dataset, dict):
                if len(dataset):
                    self._if_given_dict(dataset, y, sample_domain, sample_weight)
            elif isinstance(dataset, DeepDADataset):
                self.merge(dataset, keep_weights=True, out=False)
            else:
                try:
                    X = check_array(
                        dataset,
                        allow_nd=True,
                        ensure_2d=False,
                        ensure_min_samples=False,
                    )
                    X = to_tensor(X, device)
                except TypeError:
                    raise TypeError(
                        "Invalid dataset representation. Expected a dict of the form "
                        "{'X', 'y'(optional), 'sample_domain'(optional), "
                        "'sample_weight'(optional)},"
                        " another DeepDADataset instance or "
                        "a data type convertible to a torch Tensor"
                        f"but found {type(dataset)} instead."
                    )
                if not isinstance(sample_domain, int) and not len(sample_domain):
                    sample_domain = _DEFAULT_SAMPLE_DOMAIN_
                self._true_init(X, y, sample_domain, sample_weight)

        assert (len(self.X) == len(self.y) == len(self.sample_domain)) and (
            len(self.X) == len(self.sample_weight) or not self.has_weights
        ), (
            "Every input data X should have a domain associated, a label (if there is "
            "no label, it must be represented by torch.nan) and, optionally, a weight. "
            "If you see this message, please report the issue in the "
            "original GitHub repository at https://github.com/scikit-adaptation/skada/"
        )

    def _if_given_dict(self, dataset: dict, y, sample_domain, sample_weight):
        if "X" not in dataset:
            raise KeyError("dataset represented as dict should contain 'X' key.")

        X = dataset["X"]
        y = dataset.get("y", y)
        sample_domain = dataset.get("sample_domain", sample_domain)
        sample_weight = dataset.get("sample_weight", sample_weight)

        dataset = DeepDADataset(X, y, sample_domain, sample_weight, device=self.device)
        self.merge(dataset, keep_weights=True, out=False)

    def _true_init(self, X, y, sample_domain, sample_weight):
        if isinstance(sample_domain, int):
            sample_domain = torch.full_like(X, sample_domain)
        else:
            sample_domain = check_array(
                sample_domain, allow_nd=True, ensure_2d=False, ensure_min_samples=False
            )
            sample_domain = to_tensor(sample_domain, self.device)

        if y is None or not len(y):
            y = torch.full_like(X, _NO_LABEL_, dtype=torch.float)
            has_y = torch.full_like(X, False)
        else:
            y = check_array(
                y,
                allow_nd=True,
                ensure_all_finite="allow-nan",
                ensure_2d=False,
                ensure_min_samples=False,
            )
            y = to_tensor(y, self.device)
            has_y = y != _NO_LABEL_

        if sample_weight is None or not len(sample_weight):
            sample_weight = _EMPTY_
            has_weights = False
        else:
            sample_weight = check_array(
                sample_weight, allow_nd=True, ensure_min_samples=False, ensure_2d=False
            )
            sample_weight = to_tensor(sample_weight, device=self.device)
            has_weights = bool(len(sample_weight))

        self.X = X
        self.y = y
        self.sample_domain = sample_domain
        self.sample_weight = sample_weight
        self.has_y = has_y
        self.has_weights = has_weights

    def merge(self, other: "DeepDADataset", keep_weights=False, out=True):
        if isinstance(other, DeepDADataset):
            X = torch.cat((self.X, other.X))
            y = torch.cat((self.y, other.y))
            sample_domain = torch.cat((self.sample_domain, other.sample_domain))
            if keep_weights:
                sample_weight = torch.cat((self.sample_weight, other.sample_weight))
            else:
                sample_weight = _EMPTY_
            if out:
                return DeepDADataset(X, y, sample_domain, sample_weight, X.device)
            else:
                self.X = X
                self.y = y
                self.sample_domain = sample_domain
                self.sample_weight = sample_weight
                self.has_y = torch.cat((self.has_y, other.has_y))
                self.has_weights = bool(len(self.sample_weight))
                self.device = X.device

        else:
            raise TypeError("Can only merge two instances of DeepDADataset")

    def __add__(self, other):
        if isinstance(other, dict):
            try:
                other = DeepDADataset(other)
            except KeyError:
                raise ValueError(
                    "Can only add an instance of "
                    "DeepDADataset with another one, a convertible dict or an iterable"
                    " with convertible data representation"
                )
        elif not isinstance(other, DeepDADataset):
            try:
                other = DeepDADataset(*other)
            except TypeError:
                raise ValueError(
                    "Can only add an instance of "
                    "DeepDADataset with another one, a convertible dict or an iterable"
                    " with convertible data representation"
                )

        return self.merge(other)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if not isinstance(index, (int, slice)):
            raise TypeError(
                f"DeepDADataset indices must be int or slice, not {type(index)}"
            )

        X = {"X": self.X[index], "sample_domain": self.sample_domain[index]}

        if self.has_weights:
            X["sample_weight"] = self.sample_weight[index]

        return X, self.y[index]

    def __repr__(self):
        rep = f"DeepDADataset(\n{self.X},\n\n{self.y},\n\n{self.sample_domain}"
        if self.has_weights:
            rep += f",\n\n{self.sample_weight}"
        rep += "\n    )"
        return rep

    def as_dict(self):
        dataset = {"X": self.X, "y": self.y, "sample_domain": self.sample_domain}
        if self.has_weights:
            dataset["sample_weight"] = self.sample_weight
        return dataset

    def as_arrays(self, return_weights=False):
        if self.has_weights and return_weights:
            return self.X, self.y, self.sample_domain, self.sample_weight
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
