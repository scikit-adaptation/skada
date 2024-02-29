import pytest

torch = pytest.importorskip("torch")

from skada.deep.utils import check_generator


def test_check_generator():
    assert isinstance(check_generator(None), torch.Generator)
    assert isinstance(check_generator(0), torch.Generator)
    assert isinstance(check_generator(torch.Generator()), torch.Generator)
