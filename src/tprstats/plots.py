from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linspace, meshgrid
from statsmodels.api import OLS, add_constant


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


def plot_3D(x_label, y_label, z_label, data, elev=10, azim=45):
    """Plots a 3D scatterplot with regression plane.

    Args:
        x_label: The x_label. Much match the dataframe.
        y_label: The y_label. Much match the dataframe.
        z_label: The z_label. Much match the dataframe.
        data: The dataframe.
        elev (int, optional): Elevation view angle. Defaults to 10.
        azim (int, optional): Azimuth view angle. Defaults to 45.
    """

    x = data[x_label]
    y = data[y_label]
    z = data[z_label]

    # Fit a linear regression model
    X = add_constant(pd.DataFrame({x_label: x, y_label: y}))
    fit = OLS(z, X).fit()
    z_pred = fit.predict(X)

    # Create a grid for predictions
    grid_lines = 26
    grid_x_pred = linspace(x.min(), x.max(), grid_lines)
    grid_y_pred = linspace(y.min(), y.max(), grid_lines)
    grid_x_pred, grid_y_pred = meshgrid(grid_x_pred, grid_y_pred)
    grid_xy_pred = pd.DataFrame(
        {x_label: grid_x_pred.ravel(), y_label: grid_y_pred.ravel()}
    )
    grid_z_pred = fit.predict(add_constant(grid_xy_pred)).values.reshape(
        grid_lines, grid_lines
    )

    # Create the 3D scatter plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the actual data points
    ax.scatter(x, y, z, color="black", label="Actual", s=10)
    # Plot the predicted data points
    ax.scatter(x, y, z_pred, color="red", label="Predicted", s=10)

    # Plot the regression surface
    ax.plot_surface(grid_x_pred, grid_y_pred, grid_z_pred, color="blue", alpha=0.5)

    # Add vertical lines from the horizontal plane to the data points
    for i in range(len(x)):
        ax.plot(
            [x[i], x[i]],
            [y[i], y[i]],
            [z_pred[i], z[i]],
            color="red",
            linestyle="--",
            linewidth=0.85,
        )

    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_title(f"{z_label} as a function of {x_label} and {y_label}")

    # Set the viewing angle
    ax.view_init(elev=elev, azim=azim)

    plt.legend()
    plt.show()
