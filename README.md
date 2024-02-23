# SKADA - Domain Adaptation with scikit-learn and PyTorch

[![PyPI version](https://badge.fury.io/py/skada.svg)](https://badge.fury.io/py/skada)
[![Build Status](https://github.com/scikit-adaptation/skada/actions/workflows/testing.yml/badge.svg)](https://github.com/scikit-adaptation/skada/actions)
[![Codecov Status](https://codecov.io/gh/scikit-adaptation/skada/branch/main/graph/badge.svg)](https://codecov.io/gh/scikit-adaptation/skada)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

> [!WARNING]
> This library is currently in a phase of active development. All features are subject to change without prior notice. If you are interested in collaborating, please feel free to reach out by opening an issue or starting a discussion.

SKADA is a library for domain adaptation (DA) with a scikit-learn and PyTorch/skorch
compatible API with the following features:

- DA estimators with a scikit-learn compatible API (fit, transform, predict).
- PyTorch/skorch API for deep learning DA algorithms.
- Classifier/Regressor and data Adapter DA algorithms compatible with scikit-learn pipelines.
- Compatible with scikit-learn validation loops (cross_val_score, GridSearchCV, etc).

## Implemented algorithms

The following algorithms are currently implemented.

### Domain adaptation algorithms

- Sample reweighting methods (Gaussian [1], Discriminant [2], KLIEP [3],
  DensityRatio [4])
- Sample mapping methods (CORAL [5], Optimal Transport DA OTDA [6], LinearMonge [7])
- Subspace methods (SubspaceAlignment [8], TCA [9])
- Other methods (JDOT [10], DASVM [11])

Any methods that can be cast as an adaptation of the input data can be used as a
scikit-learn transformer (Adapter) provides both a full Classifier/Regressor
estimator and an `Adapter` that can be used in a DA pipeline with
`make_da_pipeline`. Refer to the examples below and visit [the gallery](https://scikit-adaptation.github.io/auto_examples/index.html)for more details.

### Deep learning domain adaptation algorithms

- Deep Correlation alignment (DeepCORAL [12])
- Deep joint distribution optimal (DeepJDOT [13])
- Divergence minimization (MMD/DAN [14])
- Adversarial/discriminator based DA (DANN [15], CDAN [16])

### DA metrics

- Importance Weighted [17]
- Prediction entropy [18]
- Soft neighborhood density [19]
- Deep Embedded Validation (DEV) [20]


## Installation

The library is not yet available on PyPI. You can install it from the source code.


## Short examples

We provide here a few examples to illustrate the use of the library. For more
details, please refer to this [example](https://scikit-adaptation.github.io/auto_examples/plot_how_to_use_skada.html), the [quick start guide](https://scikit-adaptation.github.io/quickstart.html) and the [gallery](https://scikit-adaptation.github.io/auto_examples/index.html).

First, the DA data in the SKADA API is stored in the following format:

```python
X, y, sample_domain 
```

Where `X` is the input data, `y` is the target labels and `sample_domain` is the
domain labels (positive for source and negative for target domains). We provide
below an exmaple ho how to fit a DA estimator:

```python
from skada import CORAL

da = CORAL()
da.fit(X, y, sample_domain=sample_domain) # sample_domain passed by name

ypred = da.predict(Xt) # predict on test data
```

One can also use `Adapter` classes to create a full pipeline with DA:

```python
from skada import CORALAdapter, make_da_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = make_da_pipeline(StandardScaler(), CORALAdapter(), LogisticRegression())

pipe.fit(X, y, sample_domain=sample_domain) # sample_domain passed by name
```

Please note that for `Adapter` classes that implement sample reweighting, the 
subsequent classifier/regressor must require sample_weights as input. This is
done with the `set_fit_requires` method. For instance, with `LogisticRegression`, you
would use `LogisticRegression().set_fit_requires('sample_weight')`:

```python
from skada import GaussianReweightDensityAdapter, make_da_pipeline
pipe = make_da_pipeline(GaussianReweightDensityAdapter(),
                        LogisticRegression().set_fit_request(sample_weight=True))
```

Finally SKADA can be used for estimating cross validation scores and parameter
selection :

```python
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from skada import CORALAdapter, make_da_pipeline
from skada.model_selection import SourceTargetShuffleSplit
from skada.metrics import PredictionEntropyScorer

# make pipeline
pipe = make_da_pipeline(StandardScaler(), CORALAdapter(), LogisticRegression())

# split and score
cv = SourceTargetShuffleSplit()
scorer = PredictionEntropyScorer()

# cross val score
scores = cross_val_score(pipe, X, y, params={'sample_domain': sample_domain}, 
                         cv=cv, scoring=scorer)

# grid search
param_grid = {'coraladapter__reg': [0.1, 0.5, 0.9]}
grid_search = GridSearchCV(estimator=pipe,
                           param_grid=param_grid,
                           cv=cv, scoring=scorer)

grid_search.fit(X, y, sample_domain=sample_domain)
```

## Acknowledgements

This toolbox has been created and is maintained by the SKADA team that includes the following members:

* [Théo Gnassounou](https://tgnassou.github.io/)
* [Oleksii Kachaiev](https://kachayev.github.io/talks/)
* [Rémi Flamary](https://remi.flamary.com/)
* [Antoine Collas](https://www.antoinecollas.fr/)
* [Yanis Lalou](https://github.com/YanisLalou)
* [Antoine de Mathelin](https://scholar.google.com/citations?user=h79bffAAAAAJ&hl=fr)

## License

The library is distributed under the 3-Clause BSD license.

## References


[1] Shimodaira Hidetoshi. ["Improving predictive inference under covariate shift by weighting the log-likelihood function."](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=235723a15c86c369c99a42e7b666dfe156ad2cba) Journal of statistical planning and inference 90, no. 2 (2000): 227-244.

[2] Sugiyama Masashi, Taiji Suzuki, and Takafumi Kanamori. ["Density-ratio matching under the Bregman divergence: a unified framework of density-ratio estimation."](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=f1467208a75def8b2e52a447ab83644db66445ea) Annals of the Institute of Statistical Mathematics 64 (2012): 1009-1044.

[3] Sugiyama Masashi, Taiji Suzuki, Shinichi Nakajima, Hisashi Kashima, Paul Von Bünau, and Motoaki Kawanabe. ["Direct importance estimation for covariate shift adaptation."](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=af14e09a9f829b9f0952eac244b0ac0c8bda2ca8) Annals of the Institute of Statistical Mathematics 60 (2008): 699-746.

[4] Sugiyama Masashi, and Klaus-Robert Müller. ["Input-dependent estimation of generalization error under covariate shift."](https://web.archive.org/web/20070221112234id_/http://sugiyama-www.cs.titech.ac.jp:80/~sugi/2005/IWSIC.pdf) (2005): 249-279.

[5] Sun Baochen, Jiashi Feng, and Kate Saenko. ["Correlation alignment for unsupervised domain adaptation."](https://arxiv.org/pdf/1612.01939.pdf) Domain adaptation in computer vision applications (2017): 153-171.

[6] Courty Nicolas, Flamary Rémi, Tuia Devis, and Alain Rakotomamonjy. ["Optimal transport for domain adaptation."](https://arxiv.org/pdf/1507.00504.pdf) IEEE Trans. Pattern Anal. Mach. Intell 1, no. 1-40 (2016): 2.

[7] Flamary, R., Lounici, K., & Ferrari, A. (2019). [Concentration bounds for linear monge mapping estimation and optimal transport domain adaptation](https://arxiv.org/pdf/1905.10155.pdf). arXiv preprint arXiv:1905.10155.

[8] Fernando, B., Habrard, A., Sebban, M., & Tuytelaars, T. (2013). [Unsupervised visual domain adaptation using subspace alignment](https://openaccess.thecvf.com/content_iccv_2013/papers/Fernando_Unsupervised_Visual_Domain_2013_ICCV_paper.pdf). In Proceedings of the IEEE international conference on computer vision (pp. 2960-2967).

[9] Pan, S. J., Tsang, I. W., Kwok, J. T., & Yang, Q. (2010). [Domain adaptation via transfer component analysis](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=4823e52161ec339d4d3526099a5477321f6a9a0f). IEEE transactions on neural networks, 22(2), 199-210.

[10] Courty, N., Flamary, R., Habrard, A., & Rakotomamonjy, A. (2017). [Joint distribution optimal transportation for domain adaptation](https://proceedings.neurips.cc/paper_files/paper/2017/file/0070d23b06b1486a538c0eaa45dd167a-Paper.pdf). Advances in neural information processing systems, 30.

[11] Bruzzone, L., & Marconcini, M. (2009). [Domain adaptation problems: A DASVM classification technique and a circular validation strategy.](https://ieeexplore.ieee.org/document/4803844) IEEE transactions on pattern analysis and machine intelligence, 32(5), 770-787.

[12] Sun, B., & Saenko, K. (2016). [Deep coral: Correlation alignment for deep domain adaptation](https://arxiv.org/pdf/1607.01719.pdf). In Computer Vision–ECCV 2016 Workshops: Amsterdam, The Netherlands, October 8-10 and 15-16, 2016, Proceedings, Part III 14 (pp. 443-450). Springer International Publishing.

[13] Damodaran, B. B., Kellenberger, B., Flamary, R., Tuia, D., & Courty, N. (2018). [Deepjdot: Deep joint distribution optimal transport for unsupervised domain adaptation](https://openaccess.thecvf.com/content_ECCV_2018/papers/Bharath_Bhushan_Damodaran_DeepJDOT_Deep_Joint_ECCV_2018_paper.pdf). In Proceedings of the European conference on computer vision (ECCV) (pp. 447-463).

[14] Long, M., Cao, Y., Wang, J., & Jordan, M. (2015, June). [Learning transferable features with deep adaptation networks](https://proceedings.mlr.press/v37/long15.pdf). In International conference on machine learning (pp. 97-105). PMLR.

[15] Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., ... & Lempitsky, V. (2016). [Domain-adversarial training of neural networks](https://www.jmlr.org/papers/volume17/15-239/15-239.pdf). Journal of machine learning research, 17(59), 1-35.

[16] Long, M., Cao, Z., Wang, J., & Jordan, M. I. (2018). [Conditional adversarial domain adaptation](https://proceedings.neurips.cc/paper_files/paper/2018/file/ab88b15733f543179858600245108dd8-Paper.pdf). Advances in neural information processing systems, 31.

[17] Sugiyama, M., Krauledat, M., & Müller, K. R. (2007). [Covariate shift adaptation by importance weighted cross validation](https://www.jmlr.org/papers/volume8/sugiyama07a/sugiyama07a.pdf). Journal of Machine Learning Research, 8(5).

[18] Morerio, P., Cavazza, J., & Murino, V. (2017).[ Minimal-entropy correlation alignment for unsupervised deep domain adaptation](https://arxiv.org/pdf/1711.10288.pdf). arXiv preprint arXiv:1711.10288.

[19] Saito, K., Kim, D., Teterwak, P., Sclaroff, S., Darrell, T., & Saenko, K. (2021). [Tune it the right way: Unsupervised validation of domain adaptation via soft neighborhood density](https://openaccess.thecvf.com/content/ICCV2021/papers/Saito_Tune_It_the_Right_Way_Unsupervised_Validation_of_Domain_Adaptation_ICCV_2021_paper.pdf). In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 9184-9193).

[20] You, K., Wang, X., Long, M., & Jordan, M. (2019, May). [Towards accurate model selection in deep unsupervised domain adaptation](https://proceedings.mlr.press/v97/you19a/you19a.pdf). In International Conference on Machine Learning (pp. 7124-7133). PMLR.



