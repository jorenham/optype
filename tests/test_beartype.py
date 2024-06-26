from beartype import BeartypeConf
from beartype.claw import beartype_package


def test_beartype_package() -> None:
    beartype_package(
        'optype',
        conf=BeartypeConf(claw_is_pep526=False, is_debug=True),
    )
