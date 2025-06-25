"""
Training setup for deep DA method.
==========================================

This example illustrates the use of deep DA methods in Skada.
on a simple image classification task.
"""
# Author: Théo Gnassounou
#
# License: BSD 3-Clause
# sphinx_gallery_thumbnail_number = 4

# %%
import torch
from skorch.dataset import Dataset

from skada.datasets import load_mnist_usps
from skada.deep import DeepCoral, DeepCoralLoss
from skada.deep.base import (
    DomainAwareCriterion,
    DomainAwareModule,
    DomainBalancedDataLoader,
)
from skada.deep.modules import MNISTtoUSPSNet

# %%
# Load the image datasets
# ----------------------------------------------------------------------------

dataset = load_mnist_usps(n_classes=2, n_samples=0.5, return_dataset=True)
X, y, sample_domain = dataset.pack(
    as_sources=["mnist"], as_targets=["usps"], mask_target_labels=True
)
X_test, y_test, sample_domain_test = dataset.pack(
    as_sources=[], as_targets=["usps"], mask_target_labels=False
)

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
loss_fn = DomainAwareCriterion(torch.nn.CrossEntropyLoss(), DeepCoralLoss(), reg=reg)

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
