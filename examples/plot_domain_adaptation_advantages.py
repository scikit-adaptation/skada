"""
Domain Adaptation: Enhancing Classifier Performance Across Domains
=============================================================================

Domain Adaptation (DA) is a crucial technique in machine learning when 
dealing with classifiers trained on one domain and applied to another. 
This tutorial aims to illustrate the decline in classifier performance across domains 
and demonstrate how to enhance it using Domain Adaptation techniques.
"""  # noqa
# %%
# Step 1: Import Necessary Libraries
# ----------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from skada import make_da_pipeline
from skada import SubspaceAlignmentAdapter
from skada.datasets import fetch_diabetes

# %%
# Step 2: Loading an example Dataset
# ----------------------------------
#
# skada comes with a few standard datasets,
# like the office31 and the diabetes for classification.
#
# In the following we will use the diabetes dataset.
# To load it, we use the :func:`~skada.datasets.fetch_diabetes_dataset` function.


# Load the diabetes dataset
domain_dataset = fetch_diabetes()

# %%
# This dataset contains 6 domains: Caucasian, African American,
# Hispanic, Asian, Other, and Unknown.
# It also has 1 binary target label: whether the patient was readmitted
# to the hospital within 30 days or not.

# %%
# Step 3: Train a classifier without Domain Adaptation techniques
# ---------------------------------------------------------------


X_train, y_train, sample_domain = domain_dataset.pack_train(
    as_sources=['Caucasian', 'Hispanic', 'Asian'],
    as_targets=['AfricanAmerican']
    )

pipe = make_da_pipeline(
    StandardScaler(),
    LogisticRegression(),
)

pipe.fit(X_train, y_train, sample_domain=sample_domain)

X_test, y_test, sample_domain = domain_dataset.pack_test(as_targets=['AfricanAmerican'])
test_score = pipe.score(X_test, y_test, sample_domain=sample_domain)

print(f"Score on target domain without adaptation techniques: {test_score}")


# %%
# Step 4: Train a classifier with a Domain Adaptation technique
# -------------------------------------------------------------

X_train, y_train, sample_domain = domain_dataset.pack_train(
    as_sources=['Caucasian', 'Hispanic', 'Asian'],
    as_targets=['AfricanAmerican']
    )

pipe = make_da_pipeline(
    StandardScaler(),
    SubspaceAlignmentAdapter(random_state = 42),
    LogisticRegression(),
)

pipe.fit(X_train, y_train, sample_domain=sample_domain)

X_test, y_test, sample_domain = domain_dataset.pack_test(as_targets=['AfricanAmerican'])
test_score = pipe.score(X_test, y_test, sample_domain=sample_domain)

print(f"Score on target domain with adaptation techniques: {test_score}")


# %%
# Conclusion
# ----------
#
# This tutorial highlighted the decline in classifier performance
# when applied to a different domain.
# By employing Domain Adaptation techniques such as
# :func:`~skada.SubspaceAlignmentAdapter`,
# we demonstrated how to enhance the classifier's performance.
# Remember to carefully choose and experiment with adaptation techniques based on
# the characteristics of your specific problem.
#
# Feel free to explore and adapt these techniques for your datasets and classifier.
