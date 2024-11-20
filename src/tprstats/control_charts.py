from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd


def control_chart(mu, sig, n, alpha, data):
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
