import matplotlib.axes

import model.model
import model.entropy
import visualisation.core

def plot_ctx_entropy_mean(priming_model: model.model.PrimingModel,
                        ax: matplotlib.axes.Axes = None,
                        disable_title: bool = False):
    ylim = [ 0, model.entropy.compute_maximum_entropy(priming_model.params.num_constructions) ]

    return visualisation.core.plot_value(
        priming_model,
        "ctx_entropy_mean",
        ylim=ylim,
        title="Mean entropy across agents",
        ax=ax,
        disable_title=disable_title,
    )