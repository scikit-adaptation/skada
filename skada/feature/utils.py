# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD 3-Clause

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
