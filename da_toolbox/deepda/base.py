from abc import ABC, abstractmethod
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset

from sklearn.model_selection import train_test_split

from ..utils import register_forwards_hook


class BaseDANetwork(ABC):
    def __init__(
        self,
        base_model,
        layer_names,
        optimizer=None,
        criterion=None,
        n_epochs=100,
        batch_size=128,
    ):
        self.base_model = base_model
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.layer_names = layer_names
        self.device = next(self.base_model.parameters()).device
        self.optimizer = optimizer
        if criterion is None:
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = criterion

    @abstractmethod
    def _loss_da(self):
        pass

    def fit(self, dataset, dataset_target=None, optimizer=None):
        base_model = copy.deepcopy(self.base_model)
        # Create hook on the wanted layers output
        self.intermediate_layers = {}
        register_forwards_hook(base_model, self.intermediate_layers, self.layer_names)

        self._make_loaders(dataset)
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(base_model.parameters())

        if dataset_target is not None:
            self.da = True
            self.loader_target = DataLoader(
                dataset_target,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
            )
        else:
            self.da = False

        best_valid_loss = np.inf
        for epoch in range(self.n_epochs):
            _ = self._train(base_model)  # return train_loss
            valid_loss = self._validate(base_model)

            # patience
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.base_model_ = base_model
                waiting = 0
            else:
                waiting += 1

    def predict(self, dataset):
        base_model = self.base_model_
        base_model.eval()
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )
        y_pred_all = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(device=self.device)
                output = base_model.forward(batch_x)
                y_pred_all.append(torch.argmax(output, axis=1).cpu().numpy())

        y_pred = np.concatenate(y_pred_all)

        return y_pred

    def best_model(self):
        return self.base_model_

    def _train(self, base_model):
        # training loop
        base_model.train()
        train_loss = []

        for batch_x, batch_y in self.loader_train:
            if self.da:
                batch_target, _ = next(iter(self.loader_target))
                # TODO change strategy: right now stop an epoch
                # when there is no more target data, but if
                # source data are more numerous not the best idea
                if batch_x.shape[0] != batch_target.shape[0]:
                    break

            self.optimizer.zero_grad()

            self.batch_x = batch_x.to(device=self.device)
            self.batch_y = batch_y.to(device=self.device)

            self.output = base_model(self.batch_x)
            self.embedd = [
                self.intermediate_layers[layer_name] for layer_name in self.layer_names
            ]

            if self.da:
                self.batch_target = batch_target.to(device=self.device)

                self.output_target = base_model(self.batch_target)
                self.embedd_target = [
                    self.intermediate_layers[layer_name]
                    for layer_name in self.layer_names
                ]

                self._loss_da().backward()
                train_loss.append(self._loss_da().item())

            else:
                self._loss().backward()
                train_loss.append(self._loss().item())

            self.optimizer.step()

        return np.mean(train_loss)

    def _validate(self, base_model):
        # validation loop
        base_model.eval()
        val_loss = np.zeros(len(self.loader_val))
        y_pred_all, y_true_all = [], []
        with torch.no_grad():
            for idx_batch, (batch_x, batch_y) in enumerate(self.loader_val):
                batch_x = batch_x.to(device=self.device)
                batch_y = batch_y.to(device=self.device)
                output = base_model.forward(batch_x)
                loss = self.criterion(output, batch_y)
                val_loss[idx_batch] = loss.item()

                y_pred_all.append(torch.argmax(output, axis=1).cpu().numpy())
                y_true_all.append(batch_y.cpu().numpy())

        # y_pred = np.concatenate(y_pred_all)
        # y_true = np.concatenate(y_true_all)
        # perf = balanced_accuracy_score(y_true, y_pred)

        return np.mean(val_loss)  # , perf

    def _loss(self):
        return self.criterion(self.output, self.batch_y)

    def _make_loaders(
        self, dataset, split_strategy="random", multi_source_strategy="concatenate"
    ):
        if type(dataset) is list:
            if split_strategy == "random":
                dataset_train, dataset_val = train_test_split(dataset, test_size=0.2)
            if multi_source_strategy == "concatenate":
                self.loader_train = DataLoader(
                    ConcatDataset(dataset_train),
                    batch_size=self.batch_size,
                    shuffle=True,
                )
                self.loader_val = DataLoader(
                    ConcatDataset(dataset_val),
                    batch_size=self.batch_size,
                    shuffle=False,
                )
        else:
            n_ds = len(dataset)
            n_ds_train = int(n_ds * 0.8)
            dataset_train, dataset_val = random_split(
                dataset, lengths=[n_ds_train, n_ds - n_ds_train]
            )
            self.loader_train = DataLoader(
                dataset_train,
                batch_size=self.batch_size,
                shuffle=True,
            )
            self.loader_val = DataLoader(
                dataset_val,
                batch_size=self.batch_size,
                shuffle=False,
            )
