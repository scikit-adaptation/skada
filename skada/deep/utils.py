# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause
import numbers
from functools import partial

from skorch.utils import _identity
import torch
from torch.nn import CrossEntropyLoss
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

    # TODO: handle more cases
    # - BCEWithLogitsLoss
    # - BCELoss
    # - MSELoss
    # But first, see: https://github.com/scikit-adaptation/skada/issues/249

    # from skorch.utils import _sigmoid_then_2d, _make_2d_probs
    # if isinstance(criterion, BCEWithLogitsLoss):
    #     return _sigmoid_then_2d

    # if isinstance(criterion, BCELoss):
    #     return _make_2d_probs

    return _identity


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

    initial_centroids : torch.Tensor or None, default=None
        Initial centroids to use. If None, centroids are initialized randomly.

    random_state : int or None, default=None
        Determines random number generation for centroid initialization.

    device : str or torch.device, default='cpu'
        The device to use for computations.

    Attributes
    ----------
    cluster_centers_ : torch.Tensor of shape (n_clusters, n_features)
        Coordinates of cluster centers.

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
                 initial_centroids=None, random_state=None, device='cpu'):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.initial_centroids = initial_centroids
        self.random_state = random_state
        self.device = device

    def _init_centroids(self, X):
        with torch.no_grad():
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
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32, device=self.device)
            else:
                X = X.to(self.device)

            # Normalize X
            X = X / torch.norm(X, dim=1, keepdim=True)

            best_inertia = None
            best_centroids = None
            best_n_iter = None

            if self.initial_centroids is None:
                for _ in range(self.n_init):
                    initial_centroids = self._init_centroids(X)
                    inertia, centroids, n_iter = self._run_single_kmean(X, initial_centroids)

                    if best_inertia is None or inertia < best_inertia:
                        best_inertia = inertia
                        best_centroids = centroids
                        best_n_iter = n_iter

            else:
                initial_centroids = self.initial_centroids.to(self.device).clone()
                inertia, centroids, n_iter = self._run_single_kmean(X, initial_centroids)
                best_centroids = centroids
                best_inertia = inertia
                best_n_iter = n_iter

            self.cluster_centers_ = best_centroids
            self.inertia_ = best_inertia
            self.n_iter_ = best_n_iter + 1

            return self
    
    def _run_single_kmean(self, X, initial_centroids):
        """Run a single spherical k-means clustering.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Training instances to cluster. Expected to be normalized.
        initial_centroids : torch.Tensor of shape (n_clusters, n_features)
            Initial centroids to use. Expected to be normalized.

        Returns
        -------
        inertia : float
            Sum of squared distances of samples to their closest cluster center.
        centroids : torch.Tensor of shape (n_clusters, n_features)
            Final positions of the centroids.
        n_iter : int
            Number of iterations run.
        """
        centroids = initial_centroids
        for n_iter in range(self.max_iter):
            # Assign samples to closest centroids
            dissimilarities = self._compute_dissimilarity_matrix(X, centroids)
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
        inertia = self._compute_dissimilarity_loss(X, centroids[labels])

        return inertia, centroids, n_iter

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
        with torch.no_grad():
            check_is_fitted(self)
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32, device=self.device)
            else:
                X = X.to(self.device)

            # No need to normalize X as it is going
            # to be normalized in cosine_similarity

            dissimilarities = self._compute_dissimilarity_matrix(X, self.cluster_centers_)
            return torch.argmin(dissimilarities, dim=1)

    def _compute_dissimilarity_matrix(self, X, centroids):
        """Compute dissimilarities between points and centroids.
        It returns a matrix of shape (X.shape[0], centroids.shape[0]).

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Input data points
        centroids : torch.Tensor of shape (n_clusters, n_features)
            Cluster centroids

        Returns
        -------
        dissimilarities : torch.Tensor of shape (n_samples, n_clusters)
            Dissimilarities between points and centroids
        """
        # Compute similarities between each sample and each centroid
        # similarities: (n_samples, n_centroids)
        similarities = cosine_similarity(X.unsqueeze(1), centroids.unsqueeze(0), dim=2)

        return 0.5 * (1 - similarities)

    def _compute_dissimilarity_loss(self, X, centroids):
        """Compute dissimilarities between points and centroids.
        It returns a scalar representing the sum of dissimilarities.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Input data points
        centroids : torch.Tensor of shape (n_clusters, n_features)
            Cluster centroids

        Returns
        -------
        dissimilarity_loss : scalar
        """
        # Compute similarities between each sample and each centroid
        # similarities: (n_samples,)
        similarities = cosine_similarity(X, centroids, dim=1)
        dissimilarities = 0.5 * (1 - similarities)

        # Compute loss
        dissimilarity_loss = dissimilarities.sum().item()

        return dissimilarity_loss
