# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: BSD 3-Clause
import numbers
from functools import partial

from skorch.utils import _identity, _sigmoid_then_2d, _make_2d_probs
import torch
from torch.nn import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss


def _get_intermediate_layers(intermediate_layers, layer_name):
    def hook(model, input, output):
        intermediate_layers[layer_name] = output.flatten(start_dim=1)

    return hook


def _register_forwards_hook(module, intermediate_layers, layer_names):
    """Add hook to chosen layers.

    The hook returns the output of intermediate layers
    in order to compute the domain adaptation loss.
    """
    for layer_name, layer_module in module.named_modules():
        if layer_name in layer_names:
            layer_module.register_forward_hook(
                _get_intermediate_layers(intermediate_layers, layer_name)
            )


def check_generator(seed):
    """Turn seed into a torch.Generator instance.

    Parameters
    ----------
    seed : None, int or instance of Generator
        If seed is None, return the Generator singleton used by torch.random.
        If seed is an int, return a new Generator instance seeded with seed.
        If seed is already a Generator instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    :class:`torch:torch.Generator`
        The generator object based on `seed` parameter.
    """
    if seed is None or seed is torch.random:
        return torch.random.manual_seed(torch.Generator().seed())
    if isinstance(seed, numbers.Integral):
        return torch.random.manual_seed(seed)
    if isinstance(seed, torch.Generator):
        return seed
    raise ValueError(
        "%r cannot be used to seed a torch.Generator instance" % seed
    )


def _infer_predict_nonlinearity(net):
    """Infers the correct nonlinearity to apply for this net

    The nonlinearity is applied only when calling
    :func:`~skorch.classifier.NeuralNetClassifier.predict` or
    :func:`~skorch.classifier.NeuralNetClassifier.predict_proba`.

    """
    # Implementation: At the moment, this function "dispatches" only
    # based on the criterion, not the class of the net. We still pass
    # the whole net as input in case we want to modify this at a
    # future point in time.
    if len(net._criteria) != 1:
        # don't know which criterion to consider, don't try to guess
        return _identity

    criterion = getattr(net, net._criteria[0] + '_').base_criterion

    if isinstance(criterion, CrossEntropyLoss):
        return partial(torch.softmax, dim=-1)

    if isinstance(criterion, BCEWithLogitsLoss):
        return _sigmoid_then_2d

    if isinstance(criterion, BCELoss):
        return _make_2d_probs

    return _identity