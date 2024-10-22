# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause
import numbers
import torch
from torch.nn.functional import cosine_similarity
from sklearn.utils.validation import check_is_fitted


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


class SphericalKMeans:
    """Spherical K-Means clustering using PyTorch.

    This algorithm is similar to K-Means but uses cosine similarity
    instead of Euclidean distance.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    n_init : int, default=10
        Number of times the k-means algorithm will be run with different
        centroid seeds.

    max_iter : int, default=300
        Maximum number of iterations of the spherical k-means algorithm
        for a single run.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    centroids : torch.Tensor or None, default=None
        Initial centroids to use. If None, centroids are initialized randomly.

    random_state : int or None, default=None
        Determines random number generation for centroid initialization.

    device : str or torch.device, default='cpu'
        The device to use for computations.

    Attributes
    ----------
    cluster_centers_ : torch.Tensor of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    labels_ : torch.Tensor of shape (n_samples,)
        Labels of each point.

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.

    n_iter_ : int
        Number of iterations run.

    References
    ----------
    Hornik, K., Feinerer, I., Kober, M., & Buchta, C. (2012). Spherical k-Means Clustering. 
    Journal of Statistical Software, 50(10), 1–22. https://doi.org/10.18637/jss.v050.i10
    """

    def __init__(self, n_clusters=8, n_init=10, max_iter=300, tol=1e-4, 
                 centroids=None, random_state=None, device='cpu'):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = centroids
        self.random_state = random_state
        self.device = device

    def _init_centroids(self, X):
        # Randomly initialize centroids
        n_samples = X.shape[0]
        indices = torch.randperm(n_samples, device=self.device)[:self.n_clusters]
        centroids = X[indices]
        return centroids / torch.norm(centroids, dim=1, keepdim=True)

    def fit(self, X, y=None):
        """Compute spherical k-means clustering.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Training instances to cluster.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(self.device)

        # Normalize X
        X = X / torch.norm(X, dim=1, keepdim=True)

        best_inertia = None
        best_centroids = None
        best_n_iter = None

        for _ in range(self.n_init):
            if self.centroids is None:
                centroids = self._init_centroids(X)
            else:
                centroids = self.centroids.to(self.device)

            for n_iter in range(self.max_iter):
                # Assign samples to closest centroids
                dissimilarities = self._compute_dissimilarities(X, centroids)
                labels = torch.argmin(dissimilarities, dim=1)

                # Update centroids
                new_centroids = torch.zeros_like(centroids)
                for k in range(self.n_clusters):
                    if torch.any(labels == k):
                        new_centroids[k] = X[labels == k].sum(dim=0)

                # Check for convergence
                if torch.allclose(centroids, new_centroids, atol=self.tol):
                    break

                centroids = new_centroids

            # Compute inertia
            dissimilarities = self._compute_dissimilarities(X, centroids[labels])
            inertia = dissimilarities.sum().item()

            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_n_iter = n_iter

        self.cluster_centers_ = best_centroids
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter + 1

        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : torch.Tensor of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self)
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(self.device)

        # No need to normalize X as it is going
        # to be normalized in cosine_similarity

        dissimilarities = self._compute_dissimilarities(X, self.cluster_centers_)
        return torch.argmin(dissimilarities, dim=1)

    def _compute_dissimilarities(self, X, centroids):
        similarities = cosine_similarity(X.unsqueeze(1), centroids.unsqueeze(0), dim=2)
        return 0.5 * (1 - similarities)
