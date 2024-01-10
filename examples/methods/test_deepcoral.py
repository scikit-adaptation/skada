# %%
from skada.datasets import make_shifted_datasets
from skada.feature import DeepCoral, ToyModule
import numpy as np

# %%
# Generate concept drift dataset
# ------------------------------
n_samples = 20
dataset = make_shifted_datasets(
    n_samples_source=n_samples,
    n_samples_target=n_samples + 1,
    shift="concept_drift",
    noise=0.1,
    random_state=42,
    return_dataset=True,
)
X, y, sample_domain = dataset.pack_train(as_sources=["s"], as_targets=["t"])

# %%
# create skorch model

net = DeepCoral(ToyModule(), reg=1)

# %%
# create a dict of X and sample_domain
X_dict = {"X": X.astype(np.float32), "sample_domain": sample_domain}
net.fit(
    X_dict,
    y,
)

X_test, y_test, sample_domain_test = dataset.pack_test(as_targets=["t"])
X_dict_test = {"X": X_test.astype(np.float32), "sample_domain": sample_domain_test}
net.score(X_dict_test, y_test)
