import matplotlib.axes

import model.model
import model.entropy
import visualisation.core

from typing import Optional

def ylim_from_model(priming_model: model.model.PrimingModel):
    ylim = [ 0, model.entropy.compute_maximum_entropy(priming_model.params.num_constructions) ]

    return ylim


def plot_ctx_entropy_mean(priming_model: model.model.PrimingModel,
                        ax: Optional[matplotlib.axes.Axes] = None,
                        disable_title: bool = False):
    ylim = ylim_from_model(priming_model)

    return visualisation.core.plot_value(
        priming_model,
        "ctx_entropy_mean",
        ylim=ylim,
        title="Mean entropy across agents",
        ax=ax,
        disable_title=disable_title,
    )

def plot_ctx_entropy_per_agent(priming_model: model.model.PrimingModel,
                              disable_title: bool = False):
    maximum_entropy = model.entropy.compute_maximum_entropy(priming_model.params.num_constructions)

    return visualisation.core.plot_ratio_pass(
        priming_model,
        "ctx_entropy_per_agent",
        ylim = [0, maximum_entropy],
        baseline=maximum_entropy / 2,
        title="Evolution of preference entropy per agent",
        disable_title=disable_title
    )


def plot_ctx_entropy_for_agent(priming_model: model.model.PrimingModel,
                             ax: Optional[matplotlib.axes.Axes] = None,
                             agent_index: Optional[int] = None,
                             disable_title: bool = False):
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