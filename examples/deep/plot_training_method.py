"""
Training setup for deep DA method.
==========================================

This example illustrates the use of deep DA methods in Skada.
on a simple image classification task.
"""
# Author: Théo Gnassounou
#
# License: MIT License
# sphinx_gallery_thumbnail_number = 4

# %%
import torchvision
from torchvision import transforms
import torch
from skorch.dataset import Dataset

import matplotlib.pyplot as plt

from skada.datasets import DomainAwareDataset
from skada.deep import DeepCoral, DeepCoralLoss
from skada.deep.modules import MNISTtoUSPSNet

from skada.deep.base import (
    DomainAwareModule,
    DomainAwareCriterion,
    DomainBalancedDataLoader,
)

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
# Training parameters
# ----------------------------------------------------------------------------

max_epochs = 2
batch_size = 256
lr = 1e-3
reg = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
# Training with skorch
# ----------------------------------------------------------------------------
model = DeepCoral(
    MNISTtoUSPSNet(),
    layer_name="fc1",
    batch_size=batch_size,
    max_epochs=max_epochs,
    train_split=False,
    reg=reg,
    lr=lr,
    device=device,
)
model.fit(X, y, sample_domain=sample_domain)


# %%
# Training with skorch with dataset
# ----------------------------------------------------------------------------
X_dict = {"X": torch.tensor(X), "sample_domain": torch.tensor(sample_domain)}

# TODO create a dataset also without skorch
dataset = Dataset(X_dict, torch.tensor(y))

model = DeepCoral(
    MNISTtoUSPSNet(),
    layer_name="fc1",
    batch_size=batch_size,
    max_epochs=max_epochs,
    train_split=False,
    reg=reg,
    lr=lr,
    device=device,
)
model.fit(dataset, y=None, sample_domain=None)

# %%
# Training with torch
# ----------------------------------------------------------------------------

model = DomainAwareModule(MNISTtoUSPSNet(), layer_name="fc1").to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
dataloader = DomainBalancedDataLoader(dataset, batch_size=batch_size)
loss_fn = DomainAwareCriterion(torch.nn.CrossEntropyLoss(), DeepCoralLoss(reg=1))

# Training loop
for epoch in range(max_epochs):
    model.train()
    running_loss = 0.0
    iter = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs, labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(**inputs, is_fit=True)
        loss = loss_fn(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        iter += 1
    print("Loss:", running_loss / iter)
