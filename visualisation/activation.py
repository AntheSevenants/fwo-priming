import matplotlib.axes

import model.model
import visualisation.core


def plot_ctx_activation_mean(model: model.model.PrimingModel,
                        ax: matplotlib.axes.Axes = None,
                        disable_title: bool = False):
    return visualisation.core.plot_ratio(
        model,
        "ctx_activation_mean",
        title="Mean activation per construction across agents",
        ax=ax,
        disable_title=disable_title,
    )

def plot_ctx_activation_per_agent(model: model.model.PrimingModel,
                             disable_title: bool = False):
    return visualisation.core.plot_ratio_pass(
        model,
        "ctx_activation_per_agent",
        ylim=[0, 1],
        baseline=0.5,
        #secondary_baseline_attribute="starting_probs_per_agent",
        title="Evolution of activation per agent",
        disable_title=disable_title
    )


def plot_ctx_activation_for_agent(model: model.model.PrimingModel,
                             ax: matplotlib.axes.Axes = None,
                             agent_index: int = None,
                             disable_title: bool = False):
    visualisation.core.check_if_none("agent_index", agent_index)

    return visualisation.core.plot_ratio(
        model,
        "ctx_activation_per_agent",
        agent_filter=agent_index,
        title=f"Activation per construction for agent {agent_index}",
        ax=ax,
        disable_title=disable_title,
    )