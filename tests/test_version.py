import optype as o


def test_version() -> None:
    assert o.__version__
    assert all(map(str.isdigit, o.__version__.split(".")[:3]))
