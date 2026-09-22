"""modelroom: which local model packages exist for your model families, and which fit your machine."""

from importlib.metadata import version

# The only version source is [project] version in pyproject.toml; the package has to be
# installed (uv sync) for this to resolve, which is also how the tests run.
__version__ = version("modelroom")

__all__ = ["__version__"]
