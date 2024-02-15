
.. _sphx_glr_api_reference:

API and modules
===============

.. currentmodule:: skada.datasets

.. automodule:: skada.datasets
   :no-members:
   :no-inherited-members:

:py:mod:`skada.datasets`:

.. autosummary::
   :toctree: gen_modules/
   :template: function.rst

   make_shifted_blobs
   make_shifted_datasets
   make_dataset_from_moons_distribution
   make_variable_frequency_dataset

.. currentmodule:: skada

.. automodule:: skada
   :no-members:
   :no-inherited-members:

:py:mod:`skada`:

.. autosummary::
   :toctree: gen_modules/
   :template: function.rst

   ReweightDensity
   GaussianReweightDensity
   DiscriminatorReweightDensity
   KLIEP
   SubspaceAlignment
   TransferComponentAnalysis
   OTMapping
   EntropicOTMapping
   ClassRegularizerOTMapping
   LinearOTMapping
   CORAL
   make_da_pipeline


.. currentmodule:: skada.deep

.. automodule:: skada.deep
   :no-members:
   :no-inherited-members:

:py:mod:`skada.deep`:

.. autosummary::
   :toctree: gen_modules/
   :template: class.rst

   DeepCoralLoss
   DeepJDOTLoss
   DANLoss
   CDANLoss

.. autosummary::
   :toctree: gen_modules/
   :template: function.rst

   dan_loss
   deepcoral_loss
   deepjdot_loss
   DeepCoral
   DeepJDOT
   DANN
   CDAN

.. currentmodule:: skada.metrics

.. automodule:: skada.metrics
   :no-members:
   :no-inherited-members:

:py:mod:`skada.metrics`:

.. autosummary::
   :toctree: gen_modules/
   :template: class.rst

   SupervisedScorer
   ImportanceWeightedScorer
   PredictionEntropyScorer
   DeepEmbeddedValidation
   SoftNeighborhoodDensity
