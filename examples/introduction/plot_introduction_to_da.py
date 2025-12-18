"""
Introduction to Domain Adaptation with SKADA
====================================================

This is a basic introduction to domain adaptation (DA) using the
:mod:`skada` library. We will introduce the main concepts of DA and show how to
use SKADA to perform DA on simple datasets.
"""

# Author: Theo Gnassounou
#
# License: BSD 3-Clause
# sphinx_gallery_thumbnail_number = 1

# %% intro
# Domain Adaptation (DA)
# --------------------------
#
# Domain Adaptation (DA) is a subfield of machine learning that focuses on
# adapting models trained on a source domain to perform well on a different but
# related target domain. This is particularly useful when there is a shift in
# data distribution between the source and target domains, which can lead to
# poor performance of models trained solely on the source domain.
#
# Let's illustrate the concept of DA with a simple example. Imagine we have
# an image classification task between pears and apples. You can train a model
# to distinguish between these two classes.
#
# .. image:: ./_static/image_classif3.pdf
#    :width: 400px
#    :align: center
#
# The model will learn discriminative features to separate the two classes
# like color, shape, texture, etc. Now one can give new example of pear and apple
# and the model will be able to classify them correctly.
#
# .. image:: ./_static/image_classif4.pdf
#    :width: 400px
#    :align: center
#
# However, if we now want to classify images of pears and apples, that shifted
# from the initial dataset like quickdraws:
#
# .. image:: ./_static/image_classif5.pdf
#    :width: 400px
#    :align: center
#
# or paintings:
# .. image:: ./_static/image_classif6.pdf
#    :width: 400px
#    :align: center
#
# The model will likely fail to classify them correctly because the data
# distribution has changed significantly. Here, the features like color, shape,
# or texture that were useful for classification in the original dataset may no
# longer be effective in the new domains.
#
# .. image:: ./_static/image_shift.pdf
#    :width: 400px
#    :align: center
#
# In domain adaptation, we suppose that we have access to different domains.
# In each domain we have the same task to solve (here classify pears and apples)
# but the data distribution is different: we have a distribution shift.
# In practice, you will suppose that you have access to **source domains** where
# you have **labeled data** and a **target domain** where you have **unlabeled data**.
# The goal of DA is to leverage the labeled data from the source domains to
# learn a model that performs well on the target domain, despite the distribution
# shift.
#
# Let's dive into SKADA to illustrate the impact of distribution shift and how DA
# can help mitigate it.

# %% imports
