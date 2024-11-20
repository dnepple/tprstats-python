__all__ = [
    "lm",
    "summary",
    "model",
    "control_chart",
    "control_chart_binary",
    "plot_objective_function",
]
# read version from installed package
from importlib.metadata import version

__version__ = version("tprstats")

from .regression import lm, summary
from .model_wrapper import model
from .control_charts import control_chart, control_chart_binary
from .plot import plot_objective_function
