from numpy import mean
import pandas as pd
import matplotlib.pyplot as plt


def plot_objective_function(fn, start, stop):
    xvar = range(start, stop)
    yvar = [mean(fn(x)) for x in xvar]
    data = pd.DataFrame({"xvar": xvar, "yvar": yvar})
    plt.plot("yvar", "xvar", data=data)
    plt.xlabel("x")
    plt.ylabel("y")
