import model.model

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from typing import List, Optional, Union, Any

COLOURS = ["blue", "orange", "green", "red", "purple"]
LINE_STYLES = ["-", "--", ":", "-."]


def formatter(x, pos, scale=100):
    del pos
    return str(int(x * scale))


def check_ax(ax: Optional[matplotlib.axes.Axes] = None,
             disable_title: bool = False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    fig = ax.get_figure()

    if disable_title:
        plt.tight_layout()

    return fig, ax


def filter_for_agent(matrix: np.ndarray,
                     agent_filter: Optional[int] = None):
    # If needed, index data for a specific agent
    if agent_filter is not None:
        # 3D matrix
        dimensionality = len(matrix.shape)

        if dimensionality == 3:
            matrix = matrix[:, agent_filter, :]
        else:
            matrix = matrix[:, agent_filter]

    return matrix


def plot_value(priming_model: model.model.PrimingModel,
               attribute: str,
               ylim: List[float],
               ax: Optional[matplotlib.axes.Axes] = None,
               agent_filter: Optional[int] = None,
               title: Optional[str] = None,
               disable_title: bool = False):
    df = priming_model.datacollector.get_model_vars_dataframe()

    fig, ax = check_ax(ax, disable_title)

    value_list = np.stack(df[attribute].tolist())
    # If needed, index data for a specific agent
    value_list = filter_for_agent(value_list, agent_filter)

    ax.plot(value_list, color=COLOURS[0])
    ax.set_ylim(*ylim)

    if title is not None and not disable_title:
        ax.set_title(title)


def plot_ratio(priming_model: model.model.PrimingModel,
               attributes: Union[str, List[str]],
               ylim: List[float] = [0, 1],
               ax: Optional[matplotlib.axes.Axes] = None,
               agent_filter: Optional[int] = None,
               title: Optional[str] = None,
               disable_title: bool = False):
    if isinstance(attributes, str):
        attributes = [attributes]  # Convert single string to list for uniform processing

    if len(attributes) > len(LINE_STYLES):
        raise ValueError(f"Number of attributes cannot exceed number of line styles (= {len(LINE_STYLES)})")

    df = priming_model.datacollector.get_model_vars_dataframe()

    fig, ax = check_ax(ax, disable_title)

    for attribute_idx, attribute in enumerate(attributes):
        matrix = np.stack(df[attribute].tolist())

        # If needed, index data for a specific agent
        matrix = filter_for_agent(matrix, agent_filter)

        for i in range(matrix.shape[1]):
            ax.plot(matrix[:, i], color=COLOURS[i], linestyle=LINE_STYLES[attribute_idx])

    if title is not None and not disable_title:
        ax.set_title(title)
    
    ax.set_ylim(*ylim)
    ax.set_yticks(np.arange(ylim[0], ylim[1] + 0.1, 0.1))


def plot_ratio_pass(priming_model: model.model.PrimingModel,
                    attribute: str,
                    ylim: List[float],
                    baseline: Optional[float] = None,
                    secondary_baseline_attribute: Optional[str] = None,
                    ax: Optional[matplotlib.axes.Axes] = None,
                    title: Optional[str] = None,
                    disable_title: Optional[bool] = False):
    df = priming_model.datacollector.get_model_vars_dataframe()

    if ax is None:
        fig, axes = plt.subplots(
            nrows=1, ncols=priming_model.params.num_agents, figsize=(15, 10), sharey=True
        )
    else:
        raise ValueError(
            "Cannot do mosaic plots for this graph type. Please do not pass an axis."
        )

    matrix = np.stack(df[attribute].tolist())
    # Secondary baseline to plot in all the graphs
    if secondary_baseline_attribute is not None:
        secondary_baselines = df[secondary_baseline_attribute][0]
    else:
        secondary_baselines = None

    num_steps = matrix.shape[0]
    time_steps = np.arange(num_steps)

    num_dimensions = len(matrix.shape)

    baseline_to_plot = None
    if baseline is not None:
        # Vertical baseline which shows 0.5
        baseline_to_plot = np.full(num_steps, baseline)

    for i, _ax in enumerate(axes):
        # Plot baselines first
        if baseline_to_plot is not None:
            _ax.plot(baseline_to_plot, time_steps, color="gray",
                    alpha=0.1, linestyle="dashed")
            
        # Vertical baseline which shows secondary baseline for this agent (if defined)
        if secondary_baselines is not None:
            starting_baseline = np.full(num_steps, secondary_baselines[i][0])
            _ax.plot(starting_baseline, time_steps, color="gray", alpha=0.1)

        if num_dimensions == 3:
            _ax.plot(matrix[:, i, 0], time_steps, color="blue")
        elif num_dimensions == 2:
            _ax.plot(matrix[:, i], time_steps, color="blue")
        else:
            raise ValueError("Invalid number of dimensions")

        _ax.set_xlim(ylim)
        _ax.set_title(f"{i + 1}")
        _ax.set_xticks([])
        # ax.set_xlabel('Construction 0 usage')
        _ax.grid(True)

        # Disable ugly boxes
        for spine in _ax.spines.values():
            spine.set_visible(False)

    axes[0].set_ylabel("Time steps in the simulation")
    axes[0].invert_yaxis()

    return ax


def check_if_none(variable_name: str, value: Any):
    if value is None:
        raise ValueError(f"\"{variable_name}\" cannot be None")