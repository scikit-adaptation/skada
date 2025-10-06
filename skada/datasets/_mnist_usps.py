# Author: Théo Gnassounou
#
# License: BSD 3-Clause

# skip if torchvision not installed
try:
    import torch
    import torchvision
    from torchvision import transforms
except ImportError:
    torchvision = False

import numpy as np

from skada.datasets import DomainAwareDataset


def load_mnist_usps(
    n_samples=1,
    n_classes=10,
    return_X_y=True,
    return_dataset=False,
    train=False,
    random_state=None,
):
    """Load the MNIST & USPS datasets and return it as a DomainAwareDataset.

    Parameters
    ----------
    n_samples : float
        Percentage of samples to return. Should be between 0 and 1.
    n_classes : int
        Number of classes to keep. Default is 5.
    return_X_y : boolean, optional (default=True)
        Returns source and target dataset as a pair of (X, y) tuples (for
        the source and the target respectively). Otherwise returns tuple of
        (X, y, sample_domain) where `sample_domain` is a categorical label
        for the domain where sample is taken.
    return_dataset : boolean, optional (default=False)
        When set to `True`, the function returns
        :class:`~skada.datasets.DomainAwareDataset` object.
    train : boolean
        When set to `True`, return the train part of the datasets (i.e., more data).
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
    """
    if not torchvision:
        raise ImportError(
            "torchvision & torch are needed to use the load_mnist_usps function. "
            "It should be installed with `pip install torch torchvision`."
        )

    if n_samples < 0 or n_samples > 1:
        raise ValueError("n_samples should be between 0 and 1.")

    rng = np.random.RandomState(random_state)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    mnist_dataset = torchvision.datasets.MNIST(
        "./datasets", train=train, transform=transform, download=True
    )
    mnist_data = torch.empty((len(mnist_dataset), 1, 28, 28))
    for i in range(len(mnist_dataset)):
        mnist_data[i] = mnist_dataset[i][0]

    mnist_target = torch.tensor(mnist_dataset.targets)

    mnist_data = mnist_data[mnist_target < n_classes]
    mnist_target = mnist_target[mnist_target < n_classes]

    selected_samples = rng.choice(
        np.linspace(0, len(mnist_data) - 1, len(mnist_data)),
        int(n_samples * len(mnist_data)),
    )
    mnist_data = mnist_data[selected_samples]
    mnist_target = mnist_target[selected_samples]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Pad(6),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    usps_dataset = torchvision.datasets.USPS(
        "./datasets", train=train, transform=transform, download=True
    )
    usps_data = torch.empty((len(usps_dataset), 1, 28, 28))
    for i in range(len(usps_dataset)):
        usps_data[i] = usps_dataset[i][0]

    usps_target = torch.tensor(usps_dataset.targets)

    usps_data = usps_data[usps_target < n_classes]
    usps_target = usps_target[usps_target < n_classes]

    selected_samples = rng.choice(
        np.linspace(0, len(usps_data) - 1, len(usps_data)),
        int(n_samples * len(usps_data)),
    )
    usps_data = usps_data[selected_samples]
    usps_target = usps_target[selected_samples]

    dataset = DomainAwareDataset(
        domains=[
            (mnist_data, mnist_target, "mnist"),
            (usps_data, usps_target, "usps"),
        ]
    )

    if return_dataset:
        return dataset
    else:
        return dataset.pack(
            as_sources=["mnist"],
            as_targets=["usps"],
            return_X_y=return_X_y,
            mask_target_labels=False,
        )
