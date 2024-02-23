"""
Divergence domain adaptation methods.
==========================================

This example illustrates the DeepCoral method from [1]
on a simple image classification task.

.. [1]  Baochen Sun and Kate Saenko. Deep coral:
        Correlation alignment for deep domain
        adaptation. In ECCV Workshops, 2016.

"""
# Author: Théo Gnassounou
#
# License: MIT License
# sphinx_gallery_thumbnail_number = 4

# %%
import torchvision
from torchvision import transforms
import torch
from torch import nn
from skorch import NeuralNetClassifier

import matplotlib.pyplot as plt

from skada.datasets import DomainAwareDataset
from skada.deep import DeepCoral
from skada.deep.modules import MNISTtoUSPSNet

# %%
# Load the image datasets
# ----------------------------------------------------------------------------

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)
mnist_dataset = torchvision.datasets.MNIST(
    "./datasets", train=False, transform=transform, download=True
)
mnist_data = torch.empty((len(mnist_dataset), 1, 28, 28))
for i in range(len(mnist_dataset)):
    mnist_data[i] = mnist_dataset[i][0]

mnist_target = torch.tensor(mnist_dataset.targets)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Pad(6),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

usps_dataset = torchvision.datasets.USPS(
    "./datasets", train=False, transform=transform, download=True
)
usps_data = torch.empty((len(usps_dataset), 1, 28, 28))
for i in range(len(usps_dataset)):
    usps_data[i] = usps_dataset[i][0]

usps_target = torch.tensor(usps_dataset.targets)

# select only 5 classes
mnist_data = mnist_data[mnist_target < 5]
mnist_target = mnist_target[mnist_target < 5]
usps_data = usps_data[usps_target < 5]
usps_target = usps_target[usps_target < 5]

# %%
figure, axes = plt.subplots(2, 5, figsize=(5, 2.7))
for j in range(5):
    axes[0, j].imshow(
        mnist_data[mnist_target == j][0].squeeze(), cmap="gray", aspect="auto"
    )
    axes[0, j].axis("off")
    axes[1, j].imshow(
        usps_data[usps_target == j][0].squeeze(), cmap="gray", aspect="auto"
    )
    axes[1, j].axis("off")
    if j == 2:
        axes[0, j].set_title("MNIST")
        axes[1, j].set_title("USPS")
plt.tight_layout()
plt.show()
# %%
# Create a domain aware dataset
# ----------------------------------------------------------------------------

dataset = DomainAwareDataset()
dataset.add_domain(mnist_data, mnist_target, domain_name="mnist")
dataset.add_domain(usps_data, usps_target, domain_name="usps")

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
# Train a DeepCoral model
# ----------------------------------------------------------------------------
model = DeepCoral(
    MNISTtoUSPSNet(),
    layer_name="fc1",
    batch_size=128,
    max_epochs=5,
    train_split=False,
    reg=1,
    lr=1e-2,
)
model.fit(X, y, sample_domain=sample_domain)
model.score(X_test, y_test, sample_domain=sample_domain_test)
