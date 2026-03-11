import matplotlib.axes

import model.model
import visualisation.core

from typing import Optional, Union, List

def plot_ctx_base_rate_mean(
        data: Union[model.model.PrimingModel, List[float]],
        min_data: Optional[List[float]] = None,
        max_data: Optional[List[float]] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        disable_title: bool = False):
    """Plot the mean base rate across agents

    Args:
        data (Union[model.model.PrimingModel, List[float]]): Either a model instance or a list of values
        min_data (Optional[List[float]], optional): List of minimal values. Needs to be defined together with max_data.
        max_data (Optional[List[float]], optional): List of maximal values. Needs to be defined together with min_data.
        ax (Optional[matplotlib.axes.Axes], optional): A pre-existing axis. Pass if you are building a multi-plot. Defaults to None.
        disable_title (bool, optional): Whether to show a title for this graph. Defaults to False.

    Returns:
        matplotlib.axes.Axis: The finished graph
    """
    
    return visualisation.core.plot_ratio(
        data,
        "ctx_base_rate_mean",
        min_data=min_data,
        max_data=max_data,
        title="Mean base rate across agents",
        ax=ax,
        disable_title=disable_title,
    )

def plot_ctx_base_rate_per_agent(
        data: Union[model.model.PrimingModel, List[float]],
        disable_title: bool = False):
    """Plot the base rate evolution of each agent on a single graph.

    Args:
        data (Union[model.model.PrimingModel, List[float]]): Either a model instance or a list of values
        disable_title (bool, optional): Whether to show a title for this grpah. Defaults to False.

    Returns:
        matplotlib.axes.Axis: The finished graph
    """

    return visualisation.core.plot_ratio_pass(
        data,
        "ctx_base_rate_per_agent",
        ylim=[0, 1],
        baseline=0.5,
        # secondary_baseline_attribute="starting_base_rate_per_agent",
        title="Evolution of relative base rate per agent",
        disable_title=disable_title
    )