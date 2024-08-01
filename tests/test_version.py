import optype as opt


def test_version():
    assert opt.__version__
    assert all(map(str.isdigit, opt.__version__.split('.')[:3]))
