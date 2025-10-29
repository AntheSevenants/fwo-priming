import matplotlib.axes

import model.model
import visualisation.core


def plot_ctx_probs_mean(model: model.model.PrimingModel,
                        ax: matplotlib.axes.Axes = None,
                        disable_title: bool = False):
    return visualisation.core.plot_ratio(
        model,
        "ctx_probs_mean",
        title="Mean probability across agents",
        ax=ax,
        disable_title=disable_title,
    )

def plot_ctx_probs_per_agent(model: model.model.PrimingModel,
                             disable_title: bool = False):
    return visualisation.core.plot_ratio_pass(
        model,
        "ctx_probs_per_agent",
        title="Evolution of relative preference per agent",
        disable_title=disable_title
    )