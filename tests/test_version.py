import optype


def test_version():
    assert optype.__version__
    assert all(map(str.isdigit, optype.__version__.split('.')[:3]))
