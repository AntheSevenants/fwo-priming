import matplotlib.axes

import model.model
import model.entropy
import visualisation.core

from typing import Optional

def ylim_from_model(priming_model: model.model.PrimingModel):
    """Derive the ylim values from the model values

    Args:
        priming_model (model.model.PrimingModel): The model instance

    Returns:
        List[float]: The minimum and maximum values for entropy for these model parameters
    """
    ylim = [ 0, model.entropy.compute_maximum_entropy(priming_model.params.num_constructions) ]

    return ylim


def plot_ctx_entropy_mean(
        priming_model: model.model.PrimingModel,
        ax: Optional[matplotlib.axes.Axes] = None,
        disable_title: bool = False):
    """Plot the mean entropy across agents

    Args:
        priming_model (model.model.PrimingModel): The model instance
        ax (Optional[matplotlib.axes.Axes], optional): A pre-existing axis. Pass if you are building a multi-plot. Defaults to None.
        disable_title (bool, optional): Whether to show a title for this graph. Defaults to False.

    Returns:
        matplotlib.axes.Axis: The finished graph
    """
    ylim = ylim_from_model(priming_model)

    return visualisation.core.plot_value(
        priming_model,
        "ctx_entropy_mean",
        ylim=ylim,
        title="Mean entropy across agents",
        ax=ax,
        disable_title=disable_title,
    )

def plot_ctx_entropy_per_agent(
        priming_model: model.model.PrimingModel,
        disable_title: bool = False):
    """Plot the entropy evolution of each agent on a single graph.

    Args:
        priming_model (model.model.PrimingModel): The model instance
        disable_title (bool, optional): Whether to show a title for this graph. Defaults to False.

    Returns:
        matplotlib.axes.Axis: The finished graph
    """

    maximum_entropy = model.entropy.compute_maximum_entropy(priming_model.params.num_constructions)

    return visualisation.core.plot_ratio_pass(
        priming_model,
        "ctx_entropy_per_agent",
        ylim = [0, maximum_entropy],
        baseline=maximum_entropy / 2,
        title="Evolution of preference entropy per agent",
        disable_title=disable_title
    )


def plot_ctx_entropy_for_agent(
        priming_model: model.model.PrimingModel,
        ax: Optional[matplotlib.axes.Axes] = None,
        agent_index: Optional[int] = None,
        disable_title: bool = False):
    """Plot the entropy evolution of a single agent

    Args:
        priming_model (model.model.PrimingModel): The model instance
        ax (Optional[matplotlib.axes.Axes], optional): A pre-existing axis. Pass if you are building a multi-plot. Defaults to None.
        agent_index (Optional[int], optional): The index of the agent to filter for. Defaults to None.
        disable_title (bool, optional): Whether to show a title for this graph. Defaults to False.

    Returns:
        matplotlib.axes.Axis: The finished graph
    """

    visualisation.core.check_if_none("agent_index", agent_index)

    ylim = ylim_from_model(priming_model)

    return visualisation.core.plot_value(
        priming_model,
        "ctx_entropy_per_agent",
        ylim=ylim,
        agent_filter=agent_index,
        title=f"Preference entropy for agent {agent_index}",
        ax=ax,
        disable_title=disable_title,
    )