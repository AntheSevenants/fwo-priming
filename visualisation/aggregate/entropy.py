import matplotlib.axes
import matplotlib.figure
import visualisation.entropy
import visualisation.aggregate.core

from typing import List, Optional, Union, Tuple


def plot_entropy_range(
    data: List[float],
    x: List[str],
    num_constructions: int,
    min_data: Optional[List[float]] = None,
    max_data: Optional[List[float]] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    disable_title: bool = False,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    ylim = visualisation.entropy.ylim_from_num_constructions(num_constructions)

    return visualisation.aggregate.core.plot_aggregate_values(
        data,
        "entropy",
        x,
        min_data=min_data,
        max_data=max_data,
        ylim=ylim,
        title="Distribution of mean entropy across selected parameter range (TODO)",
        ax=ax,
        disable_title=disable_title
    )