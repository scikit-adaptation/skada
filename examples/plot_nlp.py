r"""
Sentence-Transformers + SKADA on 20 Newsgroups
=============================================

This example shows how to perform unsupervised DA
from **talk.\*** topics to **rec.\*** topics in the
*20_newsgroups* dataset.

Steps
-----

1.  Load the dataset and keep only articles whose topics are
    talk (source) or rec (target).
2.  Embed each article with Sentence-Transformers from HuggingFace
3.  Fit a SKADA pipeline: PCA → Linear OT mapping → LogisticRegression
"""

# Author: Antoine Collas <contact@antoinecollas.fr>
# License: BSD-3-Clause
# sphinx_gallery_thumbnail_number = 1

# %% imports
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from skada import LinearOTMappingAdapter, SelectSource, make_da_pipeline
from skada.model_selection import StratifiedDomainShuffleSplit
from skada.utils import source_target_merge

# -------------------------------------------------------------------------
# 1. Fetch 20 Newsgroups & define categories                           -----
print("Fetching 20 Newsgroups …")
raw = fetch_20newsgroups(subset="all", download_if_missing=True)

SRC_POS = {"talk.politics.guns", "talk.politics.misc"}
SRC_NEG = {"rec.autos", "rec.motorcycles"}
TGT_POS = {"talk.religion.misc", "talk.politics.mideast"}
TGT_NEG = {"rec.sport.baseball", "rec.sport.hockey"}


def idx(names) -> list[int]:
    """Return integer category indices for given fine names."""
    return [raw.target_names.index(c) for c in names]


src_idx = np.isin(raw.target, idx(SRC_POS | SRC_NEG))
tgt_idx = np.isin(raw.target, idx(TGT_POS | TGT_NEG))

Xs = np.array(raw.data, dtype=object)[src_idx]
Xt = np.array(raw.data, dtype=object)[tgt_idx]

ys = np.isin(raw.target[src_idx], idx(SRC_POS)).astype(int)
yt = np.isin(raw.target[tgt_idx], idx(TGT_POS)).astype(int)

sample_domain = np.concatenate([np.ones_like(ys), -2 * np.ones_like(yt)])

# Merge for SKADA API
X, y, sample_domain = source_target_merge(Xs, Xt, ys, yt, sample_domain=sample_domain)
print(f"Total samples: {len(y)}  (source={src_idx.sum()}, target={tgt_idx.sum()})")

# -------------------------------------------------------------------------
# 2. Sentence-Transformers embeddings                                 -----
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print("Encoding texts with Sentence-Transformers …")
encoder = SentenceTransformer(
    "sentence-transformers/paraphrase-MiniLM-L3-v2", device="cpu"
)
X = encoder.encode(X, batch_size=32, show_progress_bar=True)
print(f"Embeddings shape: {X.shape}")

# -------------------------------------------------------------------------
# 3. Build pipelines                                                   -----
baseline = make_da_pipeline(
    PCA(n_components=50, random_state=0),
    SelectSource(LogisticRegression()),
)

adapted = make_da_pipeline(
    PCA(n_components=50, random_state=0),
    LinearOTMappingAdapter(),
    SelectSource(LogisticRegression()),
)

# 4 ─── evaluate on the target only ------------------------------
cv = StratifiedDomainShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
scores_base = []
scores_da = []
for tr, test in cv.split(X, y, sample_domain=sample_domain):
    baseline.fit(X[tr], y[tr], sample_domain=sample_domain[tr])
    adapted.fit(X[tr], y[tr], sample_domain=sample_domain[tr])

    tgt = test[sample_domain[test] == -2]
    scores_base.append(baseline.score(X[tgt], y[tgt]))
    scores_da.append(adapted.score(X[tgt], y[tgt]))

print(f"baseline : {np.mean(scores_base):.3f} ± {np.std(scores_base):.3f}")
print(f"lin-OT   : {np.mean(scores_da)  :.3f} ± {np.std(scores_da)  :.3f}")
