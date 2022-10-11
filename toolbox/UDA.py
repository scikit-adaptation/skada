import torch
import copy

class BaseDAEstimator():

    def __init__(self):
        pass
    
    def fit(self, X, y, X_target, y_target=None):
        model = copy.deepcopy(self.model)
        Xt = self.transform_adapt(X)
        model.train(Xt, y, loss=self.loss)
        self.base_estimator_ = model

def _do_train(
    model,
    method,
    intermediate_layers,
    loader_s,
    loader_t,
    optimizer,
    criterion,
    parameters,
    epoch,
    n_epochs,
    class_weight=None,
    domain_classifier=None,
):
    # training loop
    model.train()

    device = next(model.parameters()).device

    train_loss = list()

    len_dataloader = min(len(loader_s), len(loader_t))

    for batch_x_s, batch_y_s in loader_s:
        batch_x_t, _ = next(iter(loader_t))

        if batch_x_s.shape[0] != batch_x_t.shape[0]:
            break

        # p = float(i + epoch * len_dataloader) / n_epochs / len_dataloader
        # lamb = 2.0 / (1.0 + np.exp(-10 * p)) - 1

        optimizer.zero_grad()

        batch_x_s = batch_x_s.to(device=device)
        batch_y_s = batch_y_s.to(device=device)

        batch_x_t = batch_x_t.to(device=device)

        loss = loss_method(
            model,
            method,
            intermediate_layers,
            criterion,
            batch_x_s,
            batch_y_s,
            batch_x_t,
            parameters,
            class_weight,
            domain_classifier,
        )

        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    return np.mean(train_loss)


def _validate(model, loader, criterion):
    # validation loop
    model.eval()
    device = next(model.parameters()).device

    val_loss = np.zeros(len(loader))
    y_pred_all, y_true_all = list(), list()
    with torch.no_grad():
        for idx_batch, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(device=device)
            batch_y = batch_y.to(device=device)
            output = model.forward(batch_x)

            loss = criterion(output, batch_y)
            val_loss[idx_batch] = loss.item()

            y_pred_all.append(torch.argmax(output, axis=1).cpu().numpy())
            y_true_all.append(batch_y.cpu().numpy())

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    perf = balanced_accuracy_score(y_true, y_pred)

    return np.mean(val_loss), perf


def train(
    model,
    method,
    intermediate_layers,
    loaders_s,
    loader_v,
    loader_t,
    optimizer,
    criterion,
    parameters,
    n_epochs,
    patience=None,
    class_weight=None,
    domain_classifier=None,
):
    """Training function.

    Parameters
    ----------
    model : instance of nn.Module
        The model.
    method : str
        DA method used
    intermediate_layers : dict
        dictionary of the intermediate layer outputs.
    loader_source : instance of Sampler
        The generator of EEG samples the model has to train on.
        It contains n_train samples
    loader_target : instance of Sampler
        The generator of EEG samples the model has to test on with no label.
        It contains n_test samples. The test samples are used to
        add a OT loss to train.
    loader_valid : instance of Sampler
        The generator of EEG samples the model has to validate on.
        It contains n_val samples. The validation samples are used to
        monitor the training process and to perform early stopping
    alpha : distance parameters of the loss of JDOT
    lamb : loss target parameters of the loss of JDOT
    optimizer : instance of optimizer
        The optimizer to use for training.
    n_epochs : int
        The maximum of epochs to run.
    patience : int
        The patience parameter, i.e. how long to wait for the
        validation error to go down.
    metric : None | callable
        Metric to use to evaluate performance on the training and
        validation sets. Defaults to balanced accuracy.

    Returns
    -------
    best_model : instance of nn.Module
        The model that led to the best prediction on the validation
        dataset.
    history : list of dicts
        Training history (loss, accuracy, etc.)
    """
    history = list()
    best_valid_loss = np.inf
    best_model = copy.deepcopy(model)

    print(
        "epoch \t train_loss \t valid_loss \t source_perf \t valid_perf \t target_perf"
    )
    print("-" * 80)

    for epoch in range(1, n_epochs + 1):
        for loader_s in loaders_s:
            train_loss = _do_train(
                model,
                method,
                intermediate_layers,
                loader_s,
                loader_t,
                optimizer,
                criterion,
                parameters,
                epoch,
                n_epochs,
                class_weight,
                domain_classifier,
            )

        _, source_perf, _, _, _, _ = score(model, loader_s)
        _, target_perf, _, _, _, _ = score(model, loader_t)

        valid_loss, valid_perf = _validate(model, loader_v, criterion)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "source_perf": source_perf,
                "valid_perf": valid_perf,
                "target_perf": target_perf,
            }
        )

        print(
            f"{epoch} \t {train_loss:0.4f} \t {valid_loss:0.4f} \t"
            f"{source_perf:0.4f} \t {valid_perf:0.4f} \t {target_perf:0.4f}"
        )

        if valid_loss < best_valid_loss:
            print(f"best val loss {best_valid_loss:.4f} -> {valid_loss:.4f}")
            best_valid_loss = valid_loss
            best_model = copy.deepcopy(model)
            waiting = 0
        else:
            waiting += 1

        # model early stopping
        if patience is None:
            best_model = copy.deepcopy(model)
        else:
            if waiting >= patience:
                print(f"Stop training at epoch {epoch}")
                print(f"Best val loss : {best_valid_loss:.4f}")
                break

    return best_model, history

class CORAL(BaseDAEstimator):
    """Loss CORAL
    References
    ----------
    .. [2]  Baochen Sun and Kate Saenko. Deep coral:
            Correlation alignment for deep domain
            adaptation. In ECCV Workshops, 2016.
    """
    
    def __init__(self, architecture, layer_names):
        super().__init__()
        self.architecture = architecture
        self.layer_names = layer_names
        self.intermediate_layer = {}
    
    def fit(self, X, y, X_target, y_target=None):
        base_estimator = clone(self.base_estimator)
        self.fit_adapt(X, y, X_target)  # move X to target space
        Xt = self.transform_adapt(X)
        base_estimator.fit(Xt, y)
        self.base_estimator_ = base_estimator
        
    def predict(self, X):
        base_estimator = self.base_estimator_
        return base_estimator.predict(X)

    def fit_adapt(self, X, y, X_target):
        pass
    
    def transform_adapt(self, X):
        return X

    output_s = model(batch_x_s)
    embedd_s = intermediate_layers["feature_extractor"]
    _ = model(batch_x_t)
    embedd_t = intermediate_layers["feature_extractor"]

    Cs = cov(embedd_s)
    Ct = cov(embedd_t)

    loss_coral = parameters[0] * norm_coral(Cs, Ct)
    loss_classif = criterion(output_s, batch_y_s)

    return loss_classif + loss_coral