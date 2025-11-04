import model.model

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from typing import List, Optional

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


def plot_value(model: model.model.PrimingModel,
               attribute: str,
               ylim: List[int],
               ax: matplotlib.axes.Axes = None,
               title: str = None,
               disable_title: bool = False):
    df = model.datacollector.get_model_vars_dataframe()

    fig, ax = check_ax(ax, disable_title)

    value_list = np.stack(df[attribute])
    ax.plot(value_list, color=COLOURS[0])
    ax.set_ylim(ylim)

    if title is not None and not disable_title:
        ax.set_title(title)


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
                    ylim: List[float],
                    baseline: float = None,
                    secondary_baseline_attribute: str = None,
                    ax: matplotlib.axes.Axes = None,
                    title: str = None,
                    disable_title: bool = False):
    df = model.datacollector.get_model_vars_dataframe()

    if ax is None:
        fig, axes = plt.subplots(
            nrows=1, ncols=model.params.num_agents, figsize=(15, 10), sharey=True
        )
    else:
        raise ValueError(
            "Cannot do mosaic plots for this graph type. Please do not pass an axis."
        )

    matrix = np.stack(df[attribute])
    # Secondary baseline to plot in all the graphs
    if secondary_baseline_attribute is not None:
        secondary_baselines = df[secondary_baseline_attribute][0]
    else:
        secondary_baselines = None

    num_steps = matrix.shape[0]
    time_steps = np.arange(num_steps)

    num_dimensions = len(matrix.shape)

    if baseline is not None:
        # Vertical baseline which shows 0.5
        baseline = np.full(num_steps, baseline)

    for i, ax in enumerate(axes):
        # Plot baselines first
        if baseline is not None:
            ax.plot(baseline, time_steps, color="gray",
                    alpha=0.1, linestyle="dashed")
            
        # Vertical baseline which shows secondary baseline for this agent (if defined)
        if secondary_baselines is not None:
            starting_baseline = np.full(num_steps, secondary_baselines[i][0])
            ax.plot(starting_baseline, time_steps, color="gray", alpha=0.1)

        if num_dimensions == 3:
            ax.plot(matrix[:, i, 0], time_steps, color="blue")
        elif num_dimensions == 2:
            ax.plot(matrix[:, i], time_steps, color="blue")
        else:
            raise ValueError("Invalid number of dimensions")

        ax.set_xlim(ylim)
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
