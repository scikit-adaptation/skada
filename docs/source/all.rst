
.. _sphx_glr_api_reference:

API and modules
===============

Main module  :py:mod:`skada`
---------------------------

.. currentmodule:: skada

.. automodule:: skada
   :no-members:
   :no-inherited-members:

Sample reweighting DA methods 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: gen_modules/
   :template: function.rst

   ReweightDensity
   GaussianReweightDensity
   DiscriminatorReweightDensity
   KLIEP
   MMDTarSReweight

Sample mapping and alignment DA methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: gen_modules/
   :template: function.rst

   SubspaceAlignment
   TransferComponentAnalysis
   CORAL
   OTMapping
   EntropicOTMapping
   ClassRegularizerOTMapping
   LinearOTMapping
   
Other DA methods
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: gen_modules/
   :template: function.rst

   JDOTRegressor
   DASVMClassifier

DA pipeline
^^^^^^^^^^^

.. autosummary::
   :toctree: gen_modules/
   :template: function.rst

   make_da_pipeline

Utilities
^^^^^^^^^









Deep learning DA :py:mod:`skada.feature`
----------------------------------------

.. currentmodule:: skada.feature

.. automodule:: skada.feature
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: gen_modules/
   :template: class.rst

   DeepCORAL
   DeepJDOT
   DANN
   CDAN
   DAN

.. autosummary::
   :toctree: gen_modules/
   :template: function.rst

   dan_loss
   deepcoral_loss
   deepjdot_loss




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


Model Selection :py:mod:`skada.model_selection`
--------------------------------

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
--------------------------------

.. currentmodule:: skada.datasets

.. automodule:: skada.datasets
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: gen_modules/
   :template: function.rst

   make_shifted_blobs
   make_shifted_datasets
   make_dataset_from_moons_distribution
   make_variable_frequency_dataset


