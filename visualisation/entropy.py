import matplotlib.axes
import matplotlib.figure

import model.model
import model.entropy
import visualisation.core

from typing import Optional, List, Union, Tuple


def ylim_from_num_constructions(num_constructions: int):
    """Derive the ylim values from the model values

    Args:
        num_constructions (int): How many constructions appear in the model

    Returns:
        List[float]: The minimum and maximum values for entropy for these model parameters
    """
    ylim = [0, model.entropy.compute_maximum_entropy(num_constructions)]

    return ylim


def plot_ctx_entropy_mean(
    data: Union[model.model.PrimingModel, List[float]],
    num_constructions: int,
    base_rate: bool = False,
    min_data: Optional[List[float]] = None,
    max_data: Optional[List[float]] = None,
    x_scale_factor: int = 1,
    ax: Optional[matplotlib.axes.Axes] = None,
    disable_title: bool = False,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the mean entropy across agents

    TODO update these arguments

    Args:
        data (Union[model.model.PrimingModel, List[float]]): Either a model instance or a list of values
        num_constructions (int): The number of constructions in the simulation.
        x_scale_factor (int, optional): The factor to scale the x axis ticks by. Defaults to 1.
        ax (Optional[matplotlib.axes.Axes], optional): A pre-existing axis. Pass if you are building a multi-plot. Defaults to None.
        disable_title (bool, optional): Whether to show a title for this graph. Defaults to False.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """
    ylim = ylim_from_num_constructions(num_constructions)
    infix = "" if not base_rate else "base rate "

    return visualisation.core.plot_value(
        data,
        "ctx_entropy_mean" if not base_rate else "ctx_base_rate_entropy_mean",
        min_data=min_data,
        max_data=max_data,
        ylim=ylim,
        title=f"Mean {infix}entropy across agents",
        x_scale_factor=x_scale_factor,
        ax=ax,
        disable_title=disable_title,
    )


def plot_ctx_entropy_per_agent(
    data: Union[model.model.PrimingModel, List[List[float]]],
    num_constructions: int,
    base_rate: bool = False,
    y_scale_factor: int = 1,
    disable_title: bool = False,
) -> matplotlib.figure.Figure:
    """Plot the entropy evolution of each agent on a single graph.

    Args:
        data (Union[model.model.PrimingModel, List[List[float]]]): Either a model instance or a list of values
        num_constructions (int): The number of constructions in the simulation.
        y_scale_factor (int, optional): The factor to scale the y axis ticks by. Defaults to 1.
        disable_title (bool, optional): Whether to show a title for this graph. Defaults to False.

    Returns:
        matplotlib.figure.Figure: The finished graph
    """

    maximum_entropy = model.entropy.compute_maximum_entropy(num_constructions)
    ylim = ylim_from_num_constructions(num_constructions)

    return visualisation.core.plot_ratio_pass(
        data,
        "ctx_entropy_per_agent" if not base_rate else "ctx_base_rate_entropy_per_agent",
        ylim=ylim,
        baseline=maximum_entropy / 2,
        title="Evolution of preference entropy per agent",
        y_scale_factor=y_scale_factor,
        disable_title=disable_title,
    )


def plot_ctx_entropy_for_agent(
    data: Union[model.model.PrimingModel, List[float]],
    num_constructions: int,
    base_rate: bool = False,
    x_scale_factor: int = 1,
    ax: Optional[matplotlib.axes.Axes] = None,
    agent_index: Optional[int] = None,
    disable_title: bool = False,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the entropy evolution of a single agent

    Args:
        data (Union[model.model.PrimingModel, List[float]]): Either a model instance or a list of values
        num_constructions (int): The number of constructions in the simulation.
        x_scale_factor (int, optional): The factor to scale the x axis ticks by. Defaults to 1.
        ax (Optional[matplotlib.axes.Axes], optional): A pre-existing axis. Pass if you are building a multi-plot. Defaults to None.
        agent_index (Optional[int], optional): The index of the agent to filter for. Defaults to None.
        disable_title (bool, optional): Whether to show a title for this graph. Defaults to False.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """

    visualisation.core.check_if_none("agent_index", agent_index)

    ylim = ylim_from_num_constructions(num_constructions)

    return visualisation.core.plot_value(
        data,
        "ctx_entropy_per_agent" if not base_rate else "ctx_base_rate_entropy_per_agent",
        ylim=ylim,
        agent_filter=agent_index,
        title=f"Preference entropy for agent {agent_index}",
        x_scale_factor=x_scale_factor,
        ax=ax,
        disable_title=disable_title,
    )
