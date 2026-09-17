.. skada index.rst

.. role:: raw-html(raw)
   :format: html

.. role:: brand-blue
.. role:: brand-red

:html_theme.sidebar_secondary.remove:

.. raw:: html

   <div class="bd-header-announcement__content text-center font-weight-bold" style="background-color: var(--pst-color-primary); color: white; padding: 1rem; border-radius: 8px; margin-bottom: 2rem;">
      Welcome to SKADA: Domain Adaptation compatible with scikit-learn and PyTorch!
   </div>

==========================================================
Scikit Adaptation (SKA\ :brand-blue:`D`\ :brand-red:`A`\ )
==========================================================

.. container:: lead text-center

   Real-world machine learning fails when train and test distributions don't match. 
   **SKADA** brings production-ready Unsupervised Domain Adaptation (DA) tools straight to your standard ML pipelines.

.. container:: d-flex justify-content-center gap-3 my-4

   .. button-ref:: auto_examples/index
      :color: primary
      :shadow:
      :class: btn-lg

      View Examples Gallery

   .. button-ref:: all
      :color: secondary
      :outline:
      :class: btn-lg

      API Reference

----

.. grid:: 1 2 2 2
    :gutter: 4
    :padding: 2
    :class-container: text-center

    .. grid-item-card:: :raw-html:`<span class="skada-dot skada-dot-blue"></span>` Scikit-Learn Ecosystem
        :shadow: md

        Full ``fit``, ``transform``, and ``predict`` compatibility. Works natively inside scikit-learn pipelines, ``GridSearchCV``, and validation loops.

    .. grid-item-card:: :raw-html:`<span class="skada-dot skada-dot-red"></span>` DA Estimators
        :shadow: md

        Ready-to-use sample reweighting solvers, mapping/alignment solvers, and subspace alignment solvers.

    .. grid-item-card:: :raw-html:`<span class="skada-dot skada-dot-red"></span>` Deep Learning Methods
        :shadow: md

        Includes native ``pytorch`` and ``skorch`` support for deep domain adaptation algorithms using divergence minimization and adversarial training.

    .. grid-item-card:: :raw-html:`<span class="skada-dot skada-dot-blue"></span>` Realistic Validation
        :shadow: md

        Comes paired with specialized metrics explicitly built for realistic unsupervised model selection.

.. Toctree is kept hidden from the visual homepage but preserves structural navigation
.. toctree::
   :maxdepth: 2
   :hidden:

   Install <install>
   User Guide <auto_examples/plot_quick_start_guide>
   Examples <auto_examples/index>
   API Reference <all>
   Metrics <scorer>
   Contributing <contributing>
   Contributors <contributors>
   Release Notes <releases>