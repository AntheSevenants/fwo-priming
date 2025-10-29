import visualisation.core


def plot_ctx_probs_mean(model, ax=None):
    return visualisation.core.plot_ratio(
        model, "ctx_probs_mean", title="Mean probability across agents", ax=ax
    )
