import matplotlib.axes
import matplotlib.figure

import model.model
import model.entropy
import visualisation.core

from typing import Optional, List, Union, Tuple, Any


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
    attributes: str | List[str],
    num_constructions: int,
    is_base_rate: bool = False,
    **kwargs: Any,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the mean entropy across agents

    Args:
        data (Union[model.model.PrimingModel, List[float]]): Either a model instance or a list of values.
        attributes (str | List[str]): The column to fetch data from. Always supply, even if input data is not a model, so dimensionality of the data can be assessed.
        num_constructions (int): The number of constructions in the simulation.
        is_base_rate (bool, optional): Whether the entropy measures are derived from the base rate. Defaults to False.
        **kwargs: Additional keyword arguments passed to parent plotting function.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """
    ylim = ylim_from_num_constructions(num_constructions)
    infix = "" if not is_base_rate else "base rate "

    return visualisation.core.plot_value(
        data,
        attributes,
        ylim=ylim,
        title=f"Mean {infix}entropy across agents",
        **kwargs
    )


def plot_ctx_entropy_per_agent(
    data: Union[model.model.PrimingModel, List[List[float]]],
    attribute: str,
    num_constructions: int,
    is_base_rate: bool = False,
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """Plot the entropy evolution of each agent on a single graph.

    Args:
        data (Union[model.model.PrimingModel, List[List[float]]]): Either a model instance or a list of values.
        attribute (str): The column to fetch data from.
        num_constructions (int): The number of constructions in the simulation.
        is_base_rate (bool, optional): Whether the entropy measures are derived from the base rate. Defaults to False.
        **kwargs: Additional keyword arguments passed to parent plotting function.

    Returns:
        matplotlib.figure.Figure: The finished graph
    """

    maximum_entropy = model.entropy.compute_maximum_entropy(num_constructions)
    ylim = ylim_from_num_constructions(num_constructions)

    return visualisation.core.plot_ratio_pass(
        data,
        attribute,
        ylim=ylim,
        baseline=maximum_entropy / 2,
        title="Evolution of preference entropy per agent",
        **kwargs
    )


def plot_ctx_entropy_for_agent(
    data: Union[model.model.PrimingModel, List[float]],
    attribute: str,
    num_constructions: int,
    is_base_rate: bool = False,
    agent_index: Optional[int] = None,
    **kwargs: Any
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the entropy evolution of a single agent

    Args:
        data (Union[model.model.PrimingModel, List[float]]): Either a model instance or a list of values.
        attribute (str): The column to fetch data from.
        num_constructions (int): The number of constructions in the simulation.
        is_base_rate (bool, optional): Whether the entropy measures are derived from the base rate. Defaults to False.
        agent_index (Optional[int], optional): The index of the agent to filter for. Defaults to None.
        **kwargs: Additional keyword arguments passed to parent plotting function.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """

    visualisation.core.check_if_none("agent_index", agent_index)

    ylim = ylim_from_num_constructions(num_constructions)

    return visualisation.core.plot_value(
        data,
        attribute,
        ylim=ylim,
        agent_filter=agent_index,
        title=f"Preference entropy for agent {agent_index}",
        **kwargs
    )
