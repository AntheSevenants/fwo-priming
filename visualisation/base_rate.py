import matplotlib.axes
import matplotlib.figure

import model.model
import visualisation.core

from typing import Optional, Union, List, Tuple, Any


def plot_ctx_base_rate_mean(
    data: Union[model.model.PrimingModel, List[List[float]]],
    **kwargs: Any,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the mean base rate across agents

    Args:
        data (Union[model.model.PrimingModel, List[List[float]]]): Either a model instance or a list of values.
        **kwargs: Additional keyword arguments passed to parent plotting function.

    Returns:
        matplotlib.figure.Figure: The finished graph
    """

    return visualisation.core.plot_ratio(
        data,
        "ctx_base_rate_mean",
        title="Mean base rate across agents",
        **kwargs
    )


def plot_ctx_base_rate_per_agent(
    data: Union[model.model.PrimingModel, List[List[List[float]]]],
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """Plot the base rate evolution of each agent on a single graph.

    Args:
        data (Union[model.model.PrimingModel, List[List[List[float]]]]): Either a model instance or a list of values.
        **kwargs: Additional keyword arguments passed to parent plotting function.

    Returns:
        matplotlib.figure.Figure: The finished graph
    """

    return visualisation.core.plot_ratio_pass(
        data,
        "ctx_base_rate_per_agent",
        ylim=[0, 1],
        baseline=0.5,
        # secondary_baseline_attribute="starting_base_rate_per_agent",
        title="Evolution of relative base rate per agent",
        **kwargs
    )
