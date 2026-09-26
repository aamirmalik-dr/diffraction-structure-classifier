"""Guard that the package version and the installed distribution metadata agree."""

from importlib.metadata import version

import crystalclass


def test_version_matches_distribution_metadata() -> None:
    assert crystalclass.__version__ == version("crystalclass")
