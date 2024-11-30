import optype as op


def test_version() -> None:
    assert op.__version__
    assert all(map(str.isdigit, op.__version__.split(".")[:3]))
