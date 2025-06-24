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
# Licence: BSD 3-Clause
# sphinx_gallery_thumbnail_number = 1

# %% imports
import re

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

from datasets import concatenate_datasets, load_dataset
from skada import LinearOTMappingAdapter, make_da_pipeline
from skada.model_selection import SourceTargetShuffleSplit

# %% --------------------------------------------------------------------------
# 1. Load 20 Newsgroups and talk / rect domains
talk_re = re.compile(r"^talk\.")
rec_re = re.compile(r"^rec\.")

ds = load_dataset("SetFit/20_newsgroups", split="train")


def mark_domain(example):
    label = example["label_text"]
    if talk_re.match(label):
        example["domain"] = 0  # source
    elif rec_re.match(label):
        example["domain"] = -1  # target
    else:
        example["domain"] = -2  # discard
    return example


ds = ds.map(mark_domain).filter(lambda x: x["domain"] != -2)

talk = ds.filter(lambda x: x["domain"] == 0).select(range(200))
rec = ds.filter(lambda x: x["domain"] == -1).select(range(200))
ds = concatenate_datasets([talk, rec])

texts = ds["text"]
y = np.asarray(ds["label"])
sample_domain = np.asarray(ds["domain"])

print(texts[10][:40], "…")

# %% --------------------------------------------------------------------------
# 2. Embed texts with a tiny Sentence-Transformer
encoder = SentenceTransformer("paraphrase-TinyBERT-L6-v2", device="cpu")
X = encoder.encode(texts, batch_size=32, show_progress_bar=False)
print(f"Encoded {X.shape[0]} articles with {X.shape[1]} features each.")

# %% --------------------------------------------------------------------------
# 3. Build and fit two SKADA pipelines: one with DA and one without
pipe = make_da_pipeline(
    PCA(n_components=50, random_state=0),
    LogisticRegression(max_iter=1000),
)
pipe_da = make_da_pipeline(
    PCA(n_components=50, random_state=0),
    LinearOTMappingAdapter(),
    LogisticRegression(max_iter=1000),
)

# %% --------------------------------------------------------------------------
# 4. Evaluate with a source–target split
cv = SourceTargetShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
scores = cross_validate(pipe, X, y, cv=cv, params={"sample_domain": sample_domain})
scores_da = cross_validate(
    pipe_da, X, y, cv=cv, params={"sample_domain": sample_domain}
)

print(
    f"Base accuracy: {scores['test_score'].mean():.3f}"
    f"± {scores['test_score'].std():.3f}"
)
print(
    f"DA accuracy: {scores_da['test_score'].mean():.3f}"
    f"± {scores_da['test_score'].std():.3f}"
)
