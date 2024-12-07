__all__ = ["model", "control_chart", "control_chart_binary", "_plot_actual_fitted"]
# read version from installed package
from importlib.metadata import version

__version__ = version("tprstats")

from .models import model
from .plots import control_chart, control_chart_binary, _plot_actual_fitted
