import visualisation.core


def plot_ctx_probs_mean(model, ax=None, disable_title=False):
    return visualisation.core.plot_ratio(
        model,
        "ctx_probs_mean",
        title="Mean probability across agents",
        ax=ax,
        disable_title=disable_title,
    )

def plot_ctx_probs_per_agent(model, disable_title=False):
    return visualisation.core.plot_ratio_pass(
        model,
        "ctx_probs_per_agent",
        title="Evolution of relative preference per agent",
        disable_title=disable_title
    )