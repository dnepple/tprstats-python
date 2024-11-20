import numpy as np
import random as rd
import pandas as pd
import matplotlib.pyplot as plt


def tester(x):
    rd.seed(32)
    return [rd.randint(0, x) for i in range(0, 5)]


def plot_objective_function(fn, start, stop):
    xvar = range(start, stop)
    fn_x = list(map(fn, xvar))
    yvar = list(map(np.mean, fn_x))
    data = pd.DataFrame({"xvar": xvar, "yvar": yvar})
    plt.plot("yvar", "xvar", data=data)


plot_objective_function(tester, 1, 20)

# plotObjectiveFunction <- function(fn, from, to) {
#   results <- data.frame("xvar" = from:to)
#   # fn is assumed to return a vector and the mean is taken
#   results$yvar <- sapply(X = results$xvar, FUN = function(x) {
#     mean(fn(x))
#   })

#   graphics::plot(yvar~xvar, data = results, xlab = "x", ylab = "y", pch = 16)

#   return(results)
# }
