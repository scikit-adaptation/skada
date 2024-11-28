# Releases

## Skada v0.4.0

This update brings significant enhancements and new features:
1. New Shallow Methods: MongeAlignment and JCPOT
2. New Deep Methods: CAN, MCC, MDD, SPA, SourceOnly, and TargetOnly models.
3. Scorers: Introduced MixValScorer and improved scorer compatibility with deep models.
4. Subsampling Transformers: Added StratifiedDomainSubsampler and DomainSubsampler.
5. Deep Models: Enhanced batch handling, fixed predict_proba, stabilized MDD loss, and fixed Deep Coral.
6. Docs & Design: Added a contributor guide, new logo, and documentation updates.

### What's Changed
* Update README.md with zenodo badge by @rflamary in https://github.com/scikit-adaptation/skada/pull/216
* [MRG] Add multi-domain Monge alignment and JCPOT Target shift method by @rflamary in https://github.com/scikit-adaptation/skada/pull/180
* [MRG] Add a parameter base_criterion to deep models by @tgnassou in https://github.com/scikit-adaptation/skada/pull/217
* [MRG] Add new scorer: MixValScorer by @YanisLalou in https://github.com/scikit-adaptation/skada/pull/221
* [MRG] Fix mixval by @antoinecollas in https://github.com/scikit-adaptation/skada/pull/222
* [MRG] Fix batch issue when generating features + add sample_weight in deep models by @YanisLalou in https://github.com/scikit-adaptation/skada/pull/220
* [MRG] Allow model selection cv to handle nd inputs by @YanisLalou in https://github.com/scikit-adaptation/skada/pull/225
* [MRG] In DEV, reshape features to 2D instead of input by @YanisLalou in https://github.com/scikit-adaptation/skada/pull/226
* [MRG] Add utilities functions to the doc by @antoinecollas in https://github.com/scikit-adaptation/skada/pull/227
* Add new logo! by @tgnassou in https://github.com/scikit-adaptation/skada/pull/223
* Fix ImportanceWeightedScorer compatibility with deep learning models by @YanisLalou in https://github.com/scikit-adaptation/skada/pull/232
* [MRG] fix param for Deepjdot by @tgnassou in https://github.com/scikit-adaptation/skada/pull/234
* [MRG] Add SourceOnly and TargetOnly models by @tgnassou in https://github.com/scikit-adaptation/skada/pull/233
* [MRG] Fix docstring for the regulariation parameter of DA loss by @tgnassou in https://github.com/scikit-adaptation/skada/pull/230
* [MRG] Fix order of feature acquisition for deep module by @tgnassou in https://github.com/scikit-adaptation/skada/pull/235
* [MRG] Add recentering in DeepCoral by @tgnassou in https://github.com/scikit-adaptation/skada/pull/242
* [MRG] Add DomainOnlySampler and DomainOnlyDataloader for SourceOnly ou TargetOnly deep methods by @tgnassou in https://github.com/scikit-adaptation/skada/pull/243
* [MRG] Modify sampler to take the max of the two domains by @tgnassou in https://github.com/scikit-adaptation/skada/pull/241
* Fix: Dev scorer wasn't working with SourceOnly and TargetOnly by @YanisLalou in https://github.com/scikit-adaptation/skada/pull/244
* [MRG] Fix deep coral by @antoinecollas in https://github.com/scikit-adaptation/skada/pull/246
* [MRG] Harmonize fixtures by @antoinecollas in https://github.com/scikit-adaptation/skada/pull/248
* [MRG] Bug fix when None in make_da_pipeline by @antoinecollas in https://github.com/scikit-adaptation/skada/pull/256
* [MRG] Handle edge case Mixvalscorer by @YanisLalou in https://github.com/scikit-adaptation/skada/pull/257
* [MRG] Add CAN Method by @YanisLalou in https://github.com/scikit-adaptation/skada/pull/251
* [MRG] Uncomment MMDTarSReweightAdapter tests by @YanisLalou in https://github.com/scikit-adaptation/skada/pull/260
* [MRG] Enhancements to DomainAwareNet and Scorers to handle `allow_source` arg by @YanisLalou in https://github.com/scikit-adaptation/skada/pull/258
* [MRG] Subsampling transformer by @rflamary in https://github.com/scikit-adaptation/skada/pull/259
* [MRG] Add MCC method by @tgnassou in https://github.com/scikit-adaptation/skada/pull/250
* [MRG] Fix callback issue in CAN by @YanisLalou in https://github.com/scikit-adaptation/skada/pull/265
* [MRG] fix `predict_proba` for deep method by @tgnassou in https://github.com/scikit-adaptation/skada/pull/247
* Batchnormfix2 by @antoinedemathelin in https://github.com/scikit-adaptation/skada/pull/266
* [MRG] Handle scalar sample domain by @antoinecollas in https://github.com/scikit-adaptation/skada/pull/267
* [MRG] Add `DomainAndLabelStratifiedSubsampleTransformer` + Fix `DomainStratifiedSubsampleTransformer` by @YanisLalou in https://github.com/scikit-adaptation/skada/pull/268
* [MRG] Check if sample_domain have only unique domains indexes in check_*_domain by @apmellot in https://github.com/scikit-adaptation/skada/pull/261
* [MRG] Add epsilon in MCC to prevent log(0) by @YanisLalou in https://github.com/scikit-adaptation/skada/pull/270
* [MRG] Handle edge case for DAN by @YanisLalou in https://github.com/scikit-adaptation/skada/pull/271
* [MRG] Handle edge cases for CAN by @YanisLalou in https://github.com/scikit-adaptation/skada/pull/269
* [MRG] Add MDD method by @ambroiseodt in https://github.com/scikit-adaptation/skada/pull/263
* [MRG] Fix dissimilarities computations of Deep CAN by @antoinecollas in https://github.com/scikit-adaptation/skada/pull/274
* [MRG] Remove redundant centroid computation in spherical k-means by @YanisLalou in https://github.com/scikit-adaptation/skada/pull/275
* [MRG] Fix mdd loss by @antoinecollas in https://github.com/scikit-adaptation/skada/pull/277
* [MRG] Apply label smoothing to stabilize MDD by @antoinecollas in https://github.com/scikit-adaptation/skada/pull/279
* [MRG] do not try to complete when X_source is empty by @antoinecollas in https://github.com/scikit-adaptation/skada/pull/280
* [MRG] Add SPA method by @tgnassou in https://github.com/scikit-adaptation/skada/pull/276
* [MRG] Add contributor guide by @tgnassou in https://github.com/scikit-adaptation/skada/pull/282


**Full Changelog**: https://github.com/scikit-adaptation/skada/compare/0.3.0...0.4.0