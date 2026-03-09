import matplotlib.axes

import model.model
import visualisation.core

from typing import Optional

def plot_ctx_base_rate_mean(model: model.model.PrimingModel,
                        ax: Optional[matplotlib.axes.Axes] = None,
                        disable_title: bool = False):
    return visualisation.core.plot_ratio(
        model,
        "ctx_base_rate_mean",
        title="Mean base rate across agents",
        ax=ax,
        disable_title=disable_title,
    )

def plot_ctx_base_rate_per_agent(model: model.model.PrimingModel,
                             disable_title: bool = False):
    return visualisation.core.plot_ratio_pass(
        model,
        "ctx_base_rate_per_agent",
        ylim=[0, 1],
        baseline=0.5,
        secondary_baseline_attribute="starting_base_rate_per_agent",
        title="Evolution of relative base rate per agent",
        disable_title=disable_title
    )