
.. _sphx_glr_api_reference:

API and modules
===============

Main module :py:mod:`skada`
---------------------------

.. currentmodule:: skada

.. automodule:: skada
   :no-members:
   :no-inherited-members:

Sample reweighting DA methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. .. autosummary::
..    :toctree: gen_modules/
..    :template: function.rst

..    DensityReweight
..    GaussianReweight
..    DiscriminatorReweight
..    KLIEPReweight
..    NearestNeighborReweight
..    MMDTarSReweight

DAEstimators with adapters (Pipeline):
   .. autosummary::
      :toctree: gen_modules/
      :template: function.rst

      DensityReweight
      GaussianReweight
      DiscriminatorReweight
      KLIEPReweight
      NearestNeighborReweight
      MMDTarSReweight
      KMMReweight

Adapters:
   .. autosummary::
      :toctree: gen_modules/
      :template: class.rst

      DensityReweightAdapter
      GaussianReweightAdapter
      DiscriminatorReweightAdapter
      KLIEPReweightAdapter
      NearestNeighborReweightAdapter
      MMDTarSReweightAdapter
      KMMReweightAdapter


Sample mapping and alignment DA methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DAEstimators with adapters (Pipeline):
   .. autosummary::
      :toctree: gen_modules/
      :template: function.rst

      SubspaceAlignment
      TransferComponentAnalysis
      TransferJointMatching
      TransferSubspaceLearning
      CORAL
      OTMapping
      EntropicOTMapping
      ClassRegularizerOTMapping
      LinearOTMapping
      MMDLSConSMapping
      MultiLinearMongeAlignment

Adapters:
   .. autosummary::
      :toctree: gen_modules/
      :template: function.rst

      SubspaceAlignmentAdapter
      TransferComponentAnalysisAdapter
      TransferJointMatchingAdapter
      TransferSubspaceLearningAdapter
      CORALAdapter
      OTMappingAdapter
      EntropicOTMappingAdapter
      ClassRegularizerOTMappingAdapter
      LinearOTMappingAdapter
      MMDLSConSMappingAdapter
      MultiLinearMongeAlignmentAdapter


Other DA methods
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: gen_modules/
   :template: function.rst

   JDOTClassifier
   JDOTRegressor
   DASVMClassifier
   OTLabelProp
   JCPOTLabelProp

DA pipeline
^^^^^^^^^^^

.. autosummary::
   :toctree: gen_modules/
   :template: function.rst

   make_da_pipeline

.. autosummary::
   :toctree: gen_modules/
   :template: class.rst

   Shared
   PerDomain
   SelectSource
   SelectTarget
   SelectSourceTarget

Utilities
^^^^^^^^^

.. autosummary::
   :toctree: gen_modules/
   :template: function.rst

   source_target_split
   per_domain_split




Deep learning DA :py:mod:`skada.deep`:
----------------------------------------

.. currentmodule:: skada.deep

.. automodule:: skada.deep
   :no-members:
   :no-inherited-members:


Deep learning DA methods
^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: gen_modules/
   :template: function.rst

   DeepCoral
   DeepJDOT
   DAN
   DANN
   CDAN
   MCC
   CAN
   MDD
   SPA


SKADA deep learning DA losses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: gen_modules/
   :template: class.rst

   DeepCoralLoss
   DeepJDOTLoss
   DANLoss
   DANNLoss
   CDANLoss
   MCCLoss
   CANLoss
   MDDLoss
   SPALoss

Torch compatible DA losses in :py:mod:`skada.deep.losses`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: skada.deep.losses

.. automodule:: skada.deep.losses
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: gen_modules/
   :template: function.rst

   dan_loss
   deepcoral_loss
   deepjdot_loss
   mcc_loss
   cdd_loss
   gda_loss
   nap_loss


DA metrics :py:mod:`skada.metrics`
----------------------------------

.. currentmodule:: skada.metrics

.. automodule:: skada.metrics
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: gen_modules/
   :template: class.rst

   SupervisedScorer
   ImportanceWeightedScorer
   PredictionEntropyScorer
   DeepEmbeddedValidation
   SoftNeighborhoodDensity
   CircularValidation
   MixValScorer
   MaNoScorer


Model Selection :py:mod:`skada.model_selection`
-----------------------------------------------

.. currentmodule:: skada.model_selection

.. automodule:: skada.model_selection
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: gen_modules/
   :template: class.rst

   SourceTargetShuffleSplit
   DomainShuffleSplit
   StratifiedDomainShuffleSplit
   LeaveOneDomainOut


Datasets :py:mod:`skada.datasets`
---------------------------------

.. currentmodule:: skada.datasets

.. automodule:: skada.datasets
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: gen_modules/
   :template: class.rst

   DomainAwareDataset
   
.. autosummary::
   :toctree: gen_modules/
   :template: function.rst

   make_shifted_blobs
   make_shifted_datasets
   make_dataset_from_moons_distribution
   make_variable_frequency_dataset


Utilities :py:mod:`skada.utils`
--------------------------------

.. currentmodule:: skada.utils

.. automodule:: skada.utils
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: gen_modules/
   :template: function.rst

   check_X_y_domain
   extract_source_indices
   extract_domains_indices
   source_target_merge
