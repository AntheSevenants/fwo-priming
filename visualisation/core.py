import model.model

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

COLOURS = ["blue", "orange", "green", "red", "purple"]


def formatter(x, pos, scale=100):
    del pos
    return str(int(x * scale))


def check_ax(ax: matplotlib.axes.Axes = None,
             disable_title: bool = False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    fig = ax.get_figure()

    if disable_title:
        fig.tight_layout()

    return fig, ax


def plot_ratio(model: model.model.PrimingModel,
               attribute: str,
               ax: matplotlib.axes.Axes = None,
               title: str = None,
               disable_title: bool = False):
    df = model.datacollector.get_model_vars_dataframe()

    fig, ax = check_ax(ax, disable_title)

    matrix = np.stack(df[attribute])
    for i in range(matrix.shape[1]):
        ax.plot(matrix[:, i], color=COLOURS[i])

    if title is not None and not disable_title:
        ax.set_title(title)


def plot_ratio_pass(model: model.model.PrimingModel,
                    attribute: str, 
                    ax: matplotlib.axes.Axes = None,
                    title: str=None,
                    disable_title: bool=False):
    df = model.datacollector.get_model_vars_dataframe()

    if ax is None:
        fig, axes = plt.subplots(
            nrows=1, ncols=model.num_agents, figsize=(15, 10), sharey=True
        )
    else:
        raise ValueError(
            "Cannot do mosaic plots for this graph type. Please do not pass an ax."
        )

    matrix = np.stack(df[attribute])
    num_steps = matrix.shape[0]
    time_steps = np.arange(num_steps)
    # Vertical baseline which shows 0.5
    baseline = np.full(num_steps, 0.5)

    for i, ax in enumerate(axes):
        # Plot 0.5 baselien first
        ax.plot(baseline, time_steps, color="gray", alpha=0.1)

        ax.plot(matrix[:, i, 0], time_steps, color="blue")
        ax.set_title(f"{i + 1}")
        ax.set_xticks([])
        # ax.set_xlabel('Construction 0 usage')
        ax.grid(True)

        # Disable ugly boxes
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[0].set_ylabel("Time steps in the simulation")
    axes[0].invert_yaxis()

    return ax
