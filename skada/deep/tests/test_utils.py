try:
    import torch  # noqa: F401

except ImportError:
    torch = None

import pytest

from skada.deep.utils import check_generator


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed.")
def test_check_generator():
    assert isinstance(check_generator(None), torch.Generator)
    assert isinstance(check_generator(0), torch.Generator)
    assert isinstance(check_generator(torch.Generator()), torch.Generator)
