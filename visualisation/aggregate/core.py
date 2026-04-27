import matplotlib.figure
import matplotlib.axes
import matplotlib.pyplot as plt
import visualisation.core

import numpy as np
import pandas as pd

from typing import List, Optional, Union, Any, Tuple
from visualisation.core import COLOURS, LINE_STYLES


def make_aggregate_title_infix(parameter: Optional[str] = None) -> str:
    """Make the correct infix for the title of aggregate graphs

    Args:
        parameter (Optional[str], optional): The name of the parameter. Defaults to None.

    Returns:
        str: The given parameter is parameter is not None. Else, the string \"selected parameter\".
    """

    if parameter is None:
        return "selected parameter"
    else:
        return parameter


def plot_aggregate_values(
    data: Union[List[float], List[List[float]]],
    x: List[str],
    attributes: str,
    ylim: Optional[List[float]] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    min_data: Optional[List[float]] = None,
    max_data: Optional[List[float]] = None,
    title: Optional[str] = None,
    disable_title: bool = False,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot an error bar plot with values from an aggregation

    Args:
        data (Union[List[float], List[List[float]]]): A list of aggregate values
        attribute (str): Name of the parameter of which the values are being aggregated
        x (List[str]): A list of values for the X axis
        ylim (Optional[List[float]], optional): The expected range of values for y axis. Defaults to None.
        ax (Optional[matplotlib.axes.Axes], optional): A pre-existing axis. Pass if you are building a multi-plot. Defaults to None.
        min_data (Optional[List[List[float]]], optional): List of minimal values. Needs to be defined together with max_data.
        max_data (Optional[List[List[float]]], optional): List of maximal values. Needs to be defined together with min_data.
        title (Optional[str], optional): The title for the graph. Defaults to None.
        disable_title (bool, optional): Whether to show a title for this graph. Defaults to False.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """

    fig, ax = visualisation.core.check_ax(ax, disable_title)

    print(visualisation.core.get_value_lists(data, attributes))
    value_list = visualisation.core.get_value_lists(data, attributes)[0]
    _min_data, _max_data = visualisation.core.check_min_max_data(
        data, min_data, max_data
    )

    _yerr = (
        None
        if _min_data is None or _max_data is None
        else [np.abs(value_list - _min_data), np.abs(_max_data - value_list)]
    )

    ax.errorbar(
        x,
        value_list,
        yerr=_yerr,
        fmt="s",
        capsize=5,
        ecolor="lightgray",
        color="blue",
        elinewidth=1.5,
    )

    if title is not None and not disable_title:
        ax.set_title(title)

    if ylim is not None:
        ax.set_ylim(*ylim)

    output_fig = visualisation.core.get_ax_figure(ax)
    plt.close(output_fig)

    return (output_fig, ax)
