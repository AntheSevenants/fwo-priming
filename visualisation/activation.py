import matplotlib.axes
import matplotlib.figure

import model.model
import visualisation.core

from typing import Optional, Union, List, Tuple, Any


def plot_ctx_activation_mean(
    data: Union[model.model.PrimingModel, List[List[float]]],
    attributes: str | List[str] = "ctx_activation_mean",
    **kwargs: Any,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the mean activation level across agents.

    Args:
        data (Union[model.model.PrimingModel, List[List[float]]): Either a model instance or a list of values.
        attributes (str | List[str]): The column to fetch data from. Always supply, even if input data is not a model, so dimensionality of the data can be assessed. Defaults to "ctx_activation_mean".
        **kwargs: Additional keyword arguments passed to parent plotting function.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """

    return visualisation.core.plot_ratio(
        data,
        attributes,
        title="Mean activation per construction across agents",
        **kwargs
    )


def plot_ctx_activation_per_agent(
    data: Union[model.model.PrimingModel, List[List[List[float]]]],
    attribute: str = "ctx_activation_per_agent",
    **kwargs: Any
) -> matplotlib.figure.Figure:
    """Plot the activation level evolution of each agent on a single graph.

    Args:
        data (Union[model.model.PrimingModel, List[List[List[float]]]]): Either a model instance or a list of values.
        attribute (str): The column to fetch data from. Defaults to "ctx_activation_per_agent".
        **kwargs: Additional keyword arguments passed to parent plotting function.

    Returns:
        matplotlib.figure.Figure: The finished graph
    """

    return visualisation.core.plot_ratio_pass(
        data,
        attribute,
        ylim=[0, 1],
        baseline=0.5,
        # secondary_baseline_attribute="starting_probs_per_agent",
        title="Evolution of activation per agent",
        **kwargs
    )


def plot_ctx_activation_for_agent(
    data: Union[model.model.PrimingModel, List[List[float]]],
    attribute: str = "ctx_activation_per_agent",
    agent_index: Optional[int] = None,
    **kwargs: Any
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the activation level evolution of a single agent

    Args:
        data (Union[model.model.PrimingModel, List[List[float]]]): Either a model instance or a list of values.
        attribute (str): The column to fetch data from. Defaults to "ctx_activation_per_agent".
        agent_index (Optional[int], optional): The index of the agent to filter for. Defaults to None.
        **kwargs: Any

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """

    visualisation.core.check_if_none("agent_index", agent_index)

    return visualisation.core.plot_ratio(
        data,
        attribute,
        agent_filter=agent_index,
        title=f"Activation per construction for agent {agent_index}",
        **kwargs
    )
