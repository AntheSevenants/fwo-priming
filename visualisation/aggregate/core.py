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
    attribute: str,
    x: List[str],
    ylim: Optional[List[float]] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    min_data: Optional[List[float]] = None,
    max_data: Optional[List[float]] = None,
    title: Optional[str] = None,
    disable_title: bool = False,
):
    fig, ax = visualisation.core.check_ax(ax, disable_title)

    value_list = visualisation.core.get_value_lists(data, attribute)[0]
    _min_data, _max_data = visualisation.core.check_min_max_data(
        data, min_data, max_data
    )

    _yerr = (
        None
        if _min_data is None
        else [np.abs(value_list - _min_data), np.abs(_max_data - value_list)]
    )

    plt.errorbar(
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
