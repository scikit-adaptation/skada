Installation
============

SKADA is available on PyPI. You can install it directly from source:

.. code-block:: bash

   pip install skada

Optional dependencies
----------------------

Some features require extra dependencies:

.. code-block:: bash

   # Deep domain adaptation methods (PyTorch / Skorch)
   pip install "skada[deep]"

   # Subspace methods relying on PyTorch
   pip install "skada[subspace]"
