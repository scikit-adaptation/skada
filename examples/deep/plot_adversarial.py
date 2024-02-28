"""
Adversarial domain adaptation methods.
==========================================

This example illustrates the adversarial methods from
on a simple image classification task.
"""
# Author: Théo Gnassounou
#
# License: BSD 3-Clause
# sphinx_gallery_thumbnail_number = 4

# %%
from torch import nn
from skorch import NeuralNetClassifier

from skada.datasets import load_mnist_usps
from skada.deep import DANN
from skada.deep.modules import MNISTtoUSPSNet

# %%
# Load the image datasets
# ----------------------------------------------------------------------------

dataset = load_mnist_usps(return_dataset=True)
X, y, sample_domain = dataset.pack_train(as_sources=["mnist"], as_targets=["usps"])
X_test, y_test, sample_domain_test = dataset.pack_test(as_targets=["usps"])

# %%
# Train a classic model
# ----------------------------------------------------------------------------
model = NeuralNetClassifier(
    MNISTtoUSPSNet(),
    criterion=nn.CrossEntropyLoss(),
    batch_size=128,
    max_epochs=5,
    train_split=False,
    lr=1e-2,
)
model.fit(X[sample_domain > 0], y[sample_domain > 0])
model.score(X_test, y=y_test)

# %%
# Train a DANN model
# ----------------------------------------------------------------------------
model = DANN(
    MNISTtoUSPSNet(),
    layer_name="fc1",
    batch_size=128,
    max_epochs=5,
    train_split=False,
    reg=0.01,
    num_features=128,
    lr=1e-2,
)
model.fit(X, y, sample_domain=sample_domain)
model.score(X_test, y_test, sample_domain=sample_domain_test)
