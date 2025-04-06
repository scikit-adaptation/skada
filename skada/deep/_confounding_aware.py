# Author: Calvin McCarter <mccarter.calvin@gmail.com>
#
# License: BSD 3-Clause
from typing import Union

import miceforest as mf
import numpy as np
import scipy.stats as spst
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader

from skada.deep.losses import GroupedMMDLoss
from skada.deep.modules import LinearAdapter
from skada.deep.utils import EarlyStopping


class ConDoAdapterDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        S_list: np.ndarray,
        T_list: np.ndarray,
    ):
        # Each list has len n_bootstraps * bootsize, with elts shape=(n_mice_impute, d)
        # assert S_list.shape == T_list.shape
        assert S_list.shape[0] == T_list.shape[0]
        self.S_list = torch.from_numpy(S_list)
        self.T_list = torch.from_numpy(T_list)

    def __len__(self):
        return self.S_list.shape[0]

    def __getitem__(self, idx):
        # Returns a pair of (n_mice_impute, d) matrices as a single "sample"
        # We will compute the MMD between these two matrices
        # And the loss for a batch will be the sum over a batch of "samples"
        return self.S_list[idx, :, :], self.T_list[idx, :, :]

    def dtype(self):
        return self.S_list.dtype


def product_prior_float(Z_S, Z_T, bw_method="silverman"):
    n_S, zds = Z_S.shape
    n_T, zdt = Z_T.shape
    assert zds == zdt

    zskder = spst.gaussian_kde(Z_S.T, bw_method=bw_method)
    ztkder = spst.gaussian_kde(Z_T.T, bw_method=bw_method)
    P_SunderT = ztkder.pdf(Z_S.T)  # (n_S,)
    P_SunderT = P_SunderT / np.sum(P_SunderT)
    P_TunderS = zskder.pdf(Z_T.T)  # (n_T,)
    P_TunderS = P_TunderS / np.sum(P_TunderS)

    Z_test = np.concatenate([Z_S, Z_T], axis=0)  # (n_test, zd)
    P_test = np.concatenate([0.5 * P_SunderT, 0.5 * P_TunderS], axis=0)
    P_test = P_test.reshape(-1, 1)  # (n_test, 1)
    return Z_test, P_test, None


def product_prior_str(Z_S, Z_T):
    n_S, zd = Z_S.shape
    assert zd == 1
    n_T, zd = Z_T.shape
    assert zd == 1
    Z_test = np.unique(np.concatenate([Z_S, Z_T], axis=0)).reshape(
        1, -1
    )  # (1, numu_test)
    p_source_test = np.mean(Z_S == Z_test, axis=0)  # (numu_test,)
    p_target_test = np.mean(Z_T == Z_test, axis=0)  # (numu_test,)
    P_test = np.sqrt(p_source_test * p_target_test)  # (numu_test,)
    P_test = P_test / np.sum(P_test)

    Z_test = Z_test.reshape(-1, 1)  # (numu_test, 1)
    P_test = P_test.reshape(-1, 1)  # (numu_test, 1)
    encoder = OneHotEncoder(drop=None, sparse_output=False)
    encoder.fit(Z_test)
    return Z_test, P_test, encoder


def product_prior(Z_S, Z_T):
    assert Z_S.shape[1] == Z_T.shape[1]
    if Z_S.dtype.kind in {"U", "S"}:
        # str or unicode type
        assert Z_T.dtype.kind in {"U", "S"}
        assert Z_S.shape[1] == 1
        assert Z_T.shape[1] == 1
        return product_prior_str(Z_S, Z_T)
    else:
        assert Z_T.dtype.kind not in {"U", "S"}
        return product_prior_float(Z_S, Z_T)


class ConDoAdapterMMD:
    """Confounded Domain Adaptation using MMD as divergence function.

    See [TODO]_ for details.

    Parameters
    ----------
    transform_type : {'location-scale', 'affine'} (default='affine')
        Desired type of linear transformation
    use_mice_discrete_confounder : bool
        Whether to use MICE imputation when confounder is a discrete variable.
    mmd_size : int
        For each value sampled from prior, number of samples to evaluate MMD.
    n_mice_iters : int
        Number of iterations of MICE for conditional generation.
    bootstrap_fraction : float (default=1)
        Sets number of draws from prior as fraction of number of samples.
    n_bootstraps : int or None (default=None)
        Sets number of draws from prior. If None, uses bootstrap_fraction value.
    n_epochs : int
        Number of epochs of optimization.
    batch_size : int
        Number of draws from prior for each gradient step.
    learning_rate : float
        Learning rate for AdamW.
    weight_decay : float
        Weight decay  for AdamW.
    random_state : int (default=42)
        Random state for reproducibility.
    verbose : bool or int (default=0)
        Verbosity level for printing optimization status.

    References
    ----------
    .. [TODO] Calvin McCarter. Towards backwards-compatible data with
            confounded domain adaptation. In TMLR, 2024.
    """

    def __init__(
        self,
        transform_type: str = "affine",
        use_mice_discrete_confounder: bool = False,
        mmd_size: int = 20,
        n_mice_iters: int = 2,
        bootstrap_fraction: float = 1.0,
        n_bootstraps: int = None,  # if None, smallest possible given batch_size
        n_epochs: int = 5,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        random_state=42,
        verbose: Union[bool, int] = False,
    ):
        transforms = {"location-scale", "affine"}
        if transform_type not in transforms:
            raise NotImplementedError(f"transform_type {transform_type}")
        assert bootstrap_fraction <= 1
        self.transform_type = transform_type
        self.use_mice_discrete_confounder = use_mice_discrete_confounder
        self.mmd_size = mmd_size
        self.n_mice_iters = n_mice_iters
        self.bootstrap_fraction = bootstrap_fraction
        self.n_bootstraps = n_bootstraps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.verbose = verbose

    def fit(
        self,
        Xs: np.ndarray,
        Xt: np.ndarray,
        Zs: np.ndarray,
        Zt: np.ndarray,
    ):
        assert Xs.shape[0] == Zs.shape[0]
        assert Xt.shape[0] == Zt.shape[0]
        assert Zs.shape[1] == Zt.shape[1]
        ns, ds = Xs.shape
        nt, dt = Xt.shape
        n = min(Xs.shape[0], Xt.shape[0])

        assert Xs.dtype == Xt.dtype
        dtype = Xs.dtype
        rng = check_random_state(self.random_state)

        Z_test, W_test, encoder = product_prior(Zs, Zt)
        W_test = W_test.astype(dtype)
        n_test = Z_test.shape[0]
        discrete_confounder = encoder is not None
        use_mice = not discrete_confounder or self.use_mice_discrete_confounder
        if discrete_confounder:
            Z_test_ = encoder.transform(Z_test)
            Zs_ = encoder.transform(Zs)
            Zt_ = encoder.transform(Zt)
        else:
            Z_test_ = Z_test
            Zs_ = Zs
            Zt_ = Zt

        # bootsize = n_test * bootstrap_fraction sampled with replacement
        # Each is then given n_imp impute samples, so total dataset is of size
        # n_test * n_bootstraps * bootstrap_fraction * n_impute.
        bootsize = max(1, int(n * self.bootstrap_fraction))
        if self.n_bootstraps is None:
            n_bootstraps = int(np.ceil(self.batch_size / bootsize))
        else:
            n_bootstraps = self.n_bootstraps
        assert self.batch_size <= n_bootstraps * bootsize

        # Each list has len n_bootstraps * bootsize, with elts shape=(mmd_size, d)
        S_list = np.zeros((n_bootstraps * bootsize, self.mmd_size, ds), dtype=dtype)
        T_list = np.zeros((n_bootstraps * bootsize, self.mmd_size, dt), dtype=dtype)

        if use_mice:
            list_ix = 0
            for bix in range(n_bootstraps):
                Z_testixs = rng.choice(n_test, size=bootsize, p=W_test.ravel())
                bZ_test_ = Z_test_[Z_testixs, :]

                S_dataset = np.concatenate(
                    [
                        np.concatenate([Xs, Zs_], axis=1),
                        np.concatenate(
                            [np.full((bootsize, ds), np.nan), bZ_test_], axis=1
                        ),
                    ]
                )
                S_imputer = mf.ImputationKernel(
                    S_dataset,
                    datasets=self.mmd_size,
                    save_all_iterations=False,
                    random_state=self.random_state,
                )
                S_imputer.mice(self.n_mice_iters)
                S_complete = np.zeros((self.mmd_size, bootsize, ds), dtype=dtype)
                for imp in range(self.mmd_size):
                    S_complete[imp, :, :] = S_imputer.complete_data(dataset=imp)[
                        ns:, :ds
                    ]

                T_dataset = np.concatenate(
                    [
                        np.concatenate([Xt, Zt_], axis=1),
                        np.concatenate(
                            [np.full((bootsize, dt), np.nan), bZ_test_], axis=1
                        ),
                    ]
                )
                T_imputer = mf.ImputationKernel(
                    T_dataset,
                    datasets=self.mmd_size,
                    save_all_iterations=False,
                    random_state=self.random_state,
                )
                T_imputer.mice(self.n_mice_iters)
                T_complete = np.zeros((self.mmd_size, bootsize, dt), dtype=dtype)
                for imp in range(self.mmd_size):
                    T_complete[imp, :, :] = T_imputer.complete_data(dataset=imp)[
                        nt:, :dt
                    ]

                for i in range(bootsize):
                    S_list[list_ix, :, :] = S_complete[:, i, :]
                    T_list[list_ix, :, :] = T_complete[:, i, :]
                    list_ix += 1
            dataset = ConDoAdapterDataset(S_list, T_list)
            train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        else:
            # If not use_mice, we use the data directly
            Z_testixs = rng.choice(
                n_test, size=n_bootstraps * bootsize, p=W_test.ravel()
            )
            for list_ix in range(n_bootstraps * bootsize):
                i = Z_testixs[list_ix]
                (Zs_ixs,) = (Zs == Z_test[i, :]).ravel().nonzero()
                (Zt_ixs,) = (Zt == Z_test[i, :]).ravel().nonzero()
                S_list[list_ix, :, :] = Xs[rng.choice(Zs_ixs, size=self.mmd_size), :]
                T_list[list_ix, :, :] = Xt[rng.choice(Zt_ixs, size=self.mmd_size), :]
            dataset = ConDoAdapterDataset(S_list, T_list)
            train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model = LinearAdapter(
            transform_type=self.transform_type,
            in_features=ds,
            out_features=dt,
            dtype=dataset.dtype(),
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        early_stopping = EarlyStopping(patience=3, model=model)
        loss_fn = GroupedMMDLoss()
        n_batches = len(train_loader)
        if self.verbose:
            print(f"n_batches: {n_batches} dataset_size:{S_list.shape}")

        for epoch in range(self.n_epochs):
            model.train()
            train_loss = 0.0
            for bix, (Ssample, Tsample) in enumerate(train_loader):
                if (epoch == 0) and (bix == 0) and self.verbose:
                    print("MMD sample shapes", Ssample.shape, Tsample.shape)
                optimizer.zero_grad()
                adaptedSsample = model(Ssample)
                loss = loss_fn(adaptedSsample, Tsample)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                if self.verbose >= 2 and bix % (max(n_batches, 5) // 5) == 0:
                    # print progress ~5 times per epoch
                    tl = train_loss * n_batches / (bix + 1)
                    print(f"    epoch:{epoch} train_loss:{tl:.5f}  [{bix}/{n_batches}]")
            if self.verbose >= 2:
                print(f"    epoch:{epoch} train_loss:{train_loss:.5f}")
            if early_stopping(train_loss, model):
                break

        model.load_state_dict(early_stopping.state_dict)
        (M, b) = model.get_M_b()
        (M, b) = (M.astype(Xs.dtype), b.astype(Xs.dtype))
        if self.transform_type == "location-scale":
            self.m_ = M
            self.m_inv_ = 1 / self.m_
        elif self.transform_type == "affine":
            self.M_ = M
            if M.shape[0] == M.shape[1]:
                self.M_inv_ = np.linalg.inv(self.M_)
        self.b_ = b

    def transform(
        self,
        Xs,
    ):
        if self.transform_type == "location-scale":
            adaptedS = Xs * self.m_.reshape(1, -1) + self.b_.reshape(1, -1)
        elif self.transform_type == "affine":
            adaptedS = Xs @ self.M_.T + self.b_.reshape(1, -1)
        return adaptedS

    def inverse_transform(
        self,
        Xt,
    ):
        if self.transform_type == "location-scale":
            adaptedT = (Xt - self.b_.reshape(1, -1)) * self.m_inv_.reshape(1, -1)
        elif self.transform_type == "affine":
            adaptedT = (Xt - self.b_.reshape(1, -1)) @ self.M_inv_.T
        return adaptedT
