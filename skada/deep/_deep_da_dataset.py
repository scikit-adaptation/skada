# %%
import torch
from sklearn.utils.validation import check_array
from skorch.utils import to_tensor
from torch.utils.data import Dataset

# %%
_EMPTY_ = torch.Tensor()
_DEFAULT_SAMPLE_DOMAIN_ = 0
# I don't really like that the _NO_LABEL_ value is torch.nan,
# but torch Tensor doesn't support None
_NO_LABEL_ = torch.nan


class DeepDADataset(Dataset):
    """The Domain Aware Dataset class for deep learning.
    This class fills the gap between dictionary representation and array_like
    representation by combining them into a single object.

    All passed data will be converted to :code:`torch.Tensor`. Dict representation
    should at least contain an 'X' key containing the input data or be completely empty

    If no sample domain is provided, domain :code:`0` is attributed
    to the given data.

    If no label is provided, :code:`torch.nan` is attributed to the given data.

    When accessing an item from this dataset, returned is a dict representation
    of the element and its associated label.
    """

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
        self.has_y = self.y != _NO_LABEL_
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

        self._is_correct()

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
            sample_domain = torch.full(len(X), sample_domain)
        else:
            sample_domain = check_array(
                sample_domain, allow_nd=True, ensure_2d=False, ensure_min_samples=False
            )
            sample_domain = to_tensor(sample_domain, self.device)

        if y is None or not len(y):
            y = torch.full(len(X), _NO_LABEL_, dtype=torch.float)
            has_y = torch.full(len(X), False, dtype=torch.bool)
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

    def _is_correct(self):
        assert (len(self.X) == len(self.y) == len(self.sample_domain)) and (
            len(self.X) == len(self.sample_weight) or not self.has_weights
        ), (
            "Every input data X should have a domain associated, a label (if there is "
            "no label, it must be represented by torch.nan) and, optionally, a weight. "
        )

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

    def __truediv__(self, divisor):
        start = len(self) // divisor
        indices = torch.linspace(start, len(self), start)
        return self.split(*indices)

    def __floordiv__(self, divisor):
        return self / divisor

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

    def select(self, mask, return_weights=False):
        dataset = self.as_arrays(return_weights=return_weights)
        dataset = (data[mask] for data in dataset)
        return DeepDADataset(*dataset, device=self.device)

    def select_source(self, return_weights=False):
        mask = self.sample_domain >= 0
        return self.select(mask, return_weights)

    def select_target(self, return_weights=False):
        mask = self.sample_domain < 0
        return self.select(mask, return_weights)

    def select_domain(self, domain_id, return_weights=False):
        mask = self.sample_domain == domain_id
        return self.select(mask, return_weights)

    def per_domain_split(self, return_weights=False):
        dataset = {}
        for domain_id in self.sample_domain.unique():
            dataset[domain_id] = self.select_domain(domain_id, return_weights)
        return dataset

    def split(self, *indices):
        if any(not isinstance(index, int) for index in indices):
            raise TypeError("All splitting indices must be integers.")
        start = None
        stop = None
        res = []
        for index in indices:
            start = stop
            stop = index
            res.append(self[start:stop])
        return res

    def select_labels(self, return_weights=False):
        mask = self.has_y
        return self.select(mask, return_weights)

    def change_X(self, index, value):
        self.X[index] = value

    def change_y(self, index, value):
        self.y[index] = value

    def change_domain(self, index, value):
        self.sample_domain[index] = value

    def change_weight(self, index, value):
        if self.has_weights:
            self.sample_weight[index] = value

    def add_weights(self, sample_weight):
        self.sample_weight = sample_weight
        self.has_weights = bool(len(self.sample_weight))
        self._is_correct()

    def insert(self, indices, to_add):
        assert isinstance(indices, list) and isinstance(to_add, list), (
            "Input must be two lists or tuples representing the indices to insert at"
            " and and the datasets to insert."
        )
        to_add = (DeepDADataset(dataset) for dataset in to_add)
        split_data = self.split(*indices)
        res = DeepDADataset()
        if len(to_add) == 1:
            dataset = to_add[0]
            for subset in split_data:
                res += subset + dataset
        else:
            for original_subset, added_subset in zip(split_data, to_add):
                res += original_subset, added_subset
        return res
