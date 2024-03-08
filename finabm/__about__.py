from importlib.metadata import PackageNotFoundError, version

__version__: str | None
try:
    __version__ = version("finabm")
except PackageNotFoundError:
    __version__ = None
