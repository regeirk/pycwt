from importlib.metadata import version

import pycwt


def test_version_matches_distribution():
    assert pycwt.__version__ == version("pycwt")
    assert pycwt.version == pycwt.__version__
