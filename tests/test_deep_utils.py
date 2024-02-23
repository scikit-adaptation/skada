try:
    import torch
    from skada.deep.utils import check_generator

except ImportError:
    torch = False

import pytest


@pytest.mark.skipif(not torch, reason="PyTorch is not installed.")
def test_check_generator():
    assert isinstance(check_generator(None), torch.Generator)
    assert isinstance(check_generator(0), torch.Generator)
    assert isinstance(check_generator(torch.Generator()), torch.Generator)
