"""
How to use SKADA with SentenceTransformers
====================================================

This is a short example to perform domain adaptation (DA)
using SKADA on a natural language processing (NLP) classfication task
with SentenceTransformers from Hugging Face.
"""

# Author: Antoine Collas
#
# License: BSD 3-Clause
# sphinx_gallery_thumbnail_number = 1

# %% imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from sentence_transformers import SentenceTransformer

from skada import (
    LinearOTMappingAdapter,
    make_da_pipeline,
)
from skada.datasets import (
    fetch_amazon_review,
    get_data_home,
)
from skada.model_selection import SourceTargetShuffleSplit
from skada.utils import source_target_merge


data_home = get_data_home(None)

# %%
# Get Amazon Review dataset
# ----------
#
# We get the Amazon Review DA dataset
# are organized as follows:
#
# * :code:`X` is the input data, including the source and the target samples
# * :code:`y` is the output data to be predicted (labels on target samples are not
#   used when fitting the DA estimator)
# * :code:`sample_domain` encodes the domain of each sample (integer >=0 for
#   source and <0 for target)

# Get DA dataset
Xs, ys = fetch_amazon_review(domain="books", return_X_y=True)
Xt, yt = fetch_amazon_review(domain="dvd", return_X_y=True)
X, y, sample_domain = source_target_merge(Xs, Xt, ys, yt)

# Print a few examples of data
print(Xs[:5], ys[:5])


# %%
# Encode the data with SentenceTransformers
data_raw = data_home / "amazon_review_raw"
model = SentenceTransformer("paraphrase-TinyBERT-L6-v2", device="cpu")
X = model.encode(X, batch_size=32, show_progress_bar=True)

# %%
# Domain adaptation pipeline
pipe = make_da_pipeline(
        PCA(n_components=50),
        LinearOTMappingAdapter(),
        LogisticRegression(),
)

# %%
# Evaluation
cv = SourceTargetShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
scores = cross_validate(
    pipe,
    X,
    y,
    cv=cv,
    params={'sample_domain': sample_domain},
    scoring=PredictionEntropyScorer(),
)
