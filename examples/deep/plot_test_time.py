"""
Training setup for deep test time method.
==========================================

This example illustrates the use of deep test timeS methods in Skada.
on a simple image classification task.
"""

# Author: Marion Pavaux
#         Théo Gnassounou
#
# License: BSD 3-Clause
# sphinx_gallery_thumbnail_number = 4

# %%
import torch
from torch.utils.data import DataLoader

from skada.datasets import load_mnist_usps
from skada.deep.base import DeepDADataset
from skada.deep.losses import (
    CrossEntropyLabelSmooth,
    get_estimated_label,
    shot_full_loss,
)
from skada.deep.modules import SHOTNet

NUM_CLASSES = 2

# %%
# Load the image datasets
# ----------------------------------------------------------------------------

sub_dataset = load_mnist_usps(n_classes=NUM_CLASSES, n_samples=0.5)
dataset = DeepDADataset(*sub_dataset)
source_dataset = dataset.select_source()
target_dataset = dataset.select_target()

# %%
# Training parameters
# ----------------------------------------------------------------------------

max_epochs = 10
batch_size = 256
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
# Training with torch
# ----------------------------------------------------------------------------


model = SHOTNet(class_num=NUM_CLASSES).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
source_dataloader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
loss_fn = CrossEntropyLabelSmooth(num_classes=NUM_CLASSES)

print("Training loop")
# Training loop
for epoch in range(max_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(source_dataloader):
        inputs, labels = inputs, labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs["X"].to(device))
        loss = loss_fn(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch: {epoch} - Loss: {running_loss / batch_idx}")

target_pred = model(target_dataset.X.to(device)).cpu().detach().numpy().argmax(axis=1)
accuracy_no_adapation = (target_dataset.y.numpy() == target_pred).mean()
print(f"Accuracy on target domain without domain adaptation: {accuracy_no_adapation}")

model.classifier.eval()
optimizer = torch.optim.Adam(
    [
        {"params": model.feature_extractor.named_parameters()},
        {"params": model.bottleneck.named_parameters()},
    ],
    lr=lr,
)
dataloader_target = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)
interval_iter = 1

print("Adaptation loop")
# Adaptation loop
for epoch in range(max_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader_target:
        inputs, labels = inputs, labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        if batch_idx % interval_iter == 0:
            with torch.no_grad():
                features_full = model.bottleneck(
                    model.feature_extractor(target_dataset.X.to(device))
                )
                outputs_full = model.classifier(features_full)
            estimated_labels = get_estimated_label(outputs_full, features_full)

        # Forward pass
        outputs = model(inputs["X"].to(device))
        loss = shot_full_loss(estimated_labels[inputs["sample_idx"]], outputs)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch: {epoch} - Loss: {running_loss / batch_idx}")

target_pred = model(target_dataset.X.to(device)).cpu().detach().numpy().argmax(axis=1)
accuracy_no_adapation = (target_dataset.y.numpy() == target_pred).mean()
print(f"Accuracy on target domain with domain adaptation: {accuracy_no_adapation}")
