__all__ = [
    "lm",
    "summary",
    "model",
]
# read version from installed package
from importlib.metadata import version

__version__ = version("tprstats")

from .regression import lm, summary
from .model_wrapper import model
