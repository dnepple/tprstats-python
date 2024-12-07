from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
from numpy import arange
import statsmodels.api as sm


def control_chart(mu, sig, n, alpha, data):
    """plots a control chart for monitoring a production process for a product with a continuous measure (e.g., weight of a package).

    Args:
        mu (float): The anticipated mean when the production process is working properly.
        sig (float): The anticipated standard deviation when the process is working properly.
        n (int): The number of observations per sample.
        alpha (float): The significance level.
        data (Dataframe): The dataframe to be plotted.
    """
    ymax = mu + norm.ppf(1 - alpha / 2) * 1.1 * sig / n**0.5
    ymin = mu - norm.ppf(1 - alpha / 2) * 1.1 * sig / n**0.5
    # Calculate control limits
    upper_control_limit = mu + norm.ppf(1 - alpha / 2) * sig / n**0.5
    lower_control_limit = mu - norm.ppf(1 - alpha / 2) * sig / n**0.5

    # Create a dataframe
    df = pd.DataFrame({"data": data})

    # Plot the control chart
    plt.plot(df.index, df["data"], marker="o")
    plt.ylim(ymin, ymax)
    plt.axhline(y=mu, color="r", linestyle="-")
    plt.axhline(y=upper_control_limit, color="g", linestyle="--")
    plt.axhline(y=lower_control_limit, color="g", linestyle="--")
    plt.title("Control Chart")
    plt.xlabel("Observation Number")
    plt.ylabel("Sample Means")
    plt.show()


def control_chart_binary(p, n, alpha, data):
    """Plots a control chart for monitoring a production process for binary outcomes.

    Args:
        p (float): The target probability for the outcome expressed as a value between 0 and 1.
        n (int): The number of observations per sample._description_
        alpha (float): The significance level.
        data (Dataframe): The dataframe to be plotted.
    """
    sig = (p * (1 - p)) ** 0.5
    ymax = p + norm.ppf(1 - alpha / 2) * 1.1 * sig / n**0.5

    # Calculate control limits
    upper_control_limit = p + norm.ppf(1 - alpha / 2) * sig / n**0.5
    lower_control_limit = p - norm.ppf(1 - alpha / 2) * sig / n**0.5

    # Create a dataframe
    df = pd.DataFrame({"data": data})

    # Plot the control chart
    plt.plot(df.index, df["data"], marker="o")
    plt.ylim(0, ymax)
    plt.axhline(y=p, color="r", linestyle="-")
    plt.axhline(y=upper_control_limit, color="g", linestyle="--")
    plt.axhline(y=lower_control_limit, color="g", linestyle="--")
    plt.title("Control Chart - Binary Variable")
    plt.xlabel("Observation Number")
    plt.ylabel("Sample Means")
    plt.show()


def _plot_actual_fitted(y, y_id, predicted, upper, lower):
    Observation = arange(1, len(y) + 1)

    # Determine the y-axis limits
    ymax = max(upper + 0.5)
    ymin = min(lower - 0.5)

    # Plot actual values, predicted values, and prediction intervals
    plt.figure(figsize=(10, 6))
    plt.scatter(Observation, y, color="black", label="Actual", s=20)
    plt.plot(Observation, y, color="black")
    plt.plot(Observation, predicted, color="red", label="Predicted")
    plt.plot(Observation, upper, color="blue", label="95% PI")
    plt.plot(Observation, lower, color="blue")

    plt.ylim(ymin, ymax)
    plt.xlabel("Observation")
    plt.ylabel(y_id)
    plt.title("Actual (black), Predicted (red), and 95% PI (blue)")
    plt.legend()
    plt.show()
