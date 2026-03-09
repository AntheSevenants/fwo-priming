import matplotlib.axes

import model.model
import visualisation.core

from typing import Optional


def plot_ctx_activation_mean(
        model: model.model.PrimingModel,
        ax: Optional[matplotlib.axes.Axes] = None,
        disable_title: bool = False):
    """Plot the mean activation level across agents.

    Args:
        model (model.model.PrimingModel): The model instance
        ax (Optional[matplotlib.axes.Axes], optional): A pre-existing axis. Pass if you are building a multi-plot. Defaults to None.
        disable_title (bool, optional): Whether to show a title for this graph. Defaults to False.

    Returns:
        matplotlib.axes.Axis: The finished graph
    """

    return visualisation.core.plot_ratio(
        model,
        "ctx_activation_mean",
        title="Mean activation per construction across agents",
        ax=ax,
        disable_title=disable_title,
    )

def plot_ctx_activation_per_agent(
        model: model.model.PrimingModel,
        disable_title: bool = False):
    """Plot the activation level evolution of each agent on a single graph.

    Args:
        model (model.model.PrimingModel): The model instance
        disable_title (bool, optional): Whether to show a title for this graph. Defaults to False.

    Returns:
        matplotlib.axes.Axis: The finished graph
    """

    return visualisation.core.plot_ratio_pass(
        model,
        "ctx_activation_per_agent",
        ylim=[0, 1],
        baseline=0.5,
        #secondary_baseline_attribute="starting_probs_per_agent",
        title="Evolution of activation per agent",
        disable_title=disable_title
    )


def plot_ctx_activation_for_agent(
        model: model.model.PrimingModel,
        ax:  Optional[matplotlib.axes.Axes] = None,
        agent_index: Optional[int] = None,
        disable_title: bool = False):
    """Plot the activation level evolution of a single agent

    Args:
        model (model.model.PrimingModel): The model instance
        ax (Optional[matplotlib.axes.Axes], optional): A pre-existing axis. Pass if you are building a multi-plot. Defaults to None.
        agent_index (Optional[int], optional): The index of the agent to filter for. Defaults to None.
        disable_title (bool, optional): Whether to show a title for this graph. Defaults to False.

    Returns:
        matplotlib.axes.Axis: The finished graph
    """

    visualisation.core.check_if_none("agent_index", agent_index)

    return visualisation.core.plot_ratio(
        model,
        "ctx_activation_per_agent",
        agent_filter=agent_index,
        title=f"Activation per construction for agent {agent_index}",
        ax=ax,
        disable_title=disable_title,
    )