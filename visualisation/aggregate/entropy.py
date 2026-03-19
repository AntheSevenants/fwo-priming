import matplotlib.axes
import matplotlib.figure
import visualisation.entropy
import visualisation.aggregate.core

from typing import List, Optional, Union, Tuple


def plot_entropy_range(
    data: List[float],
    x: List[str],
    num_constructions: int,
    parameter: Optional[str] = None,
    base_rate: bool = True,
    min_data: Optional[List[float]] = None,
    max_data: Optional[List[float]] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    disable_title: bool = False,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    ylim = visualisation.entropy.ylim_from_num_constructions(num_constructions)
    title_infix = visualisation.aggregate.core.make_aggregate_title_infix(parameter)

    infix = "" if not base_rate else "base rate "

    return visualisation.aggregate.core.plot_aggregate_values(
        data,
        "entropy" if not base_rate else "base_rate_entropy",
        x,
        min_data=min_data,
        max_data=max_data,
        ylim=ylim,
        title=f"Distribution of mean {infix}entropy across {title_infix} range",
        ax=ax,
        disable_title=disable_title,
    )
