import matplotlib.figure
import matplotlib.axes

import visualisation.aggregate.core

from typing import Optional, Union, List, Tuple, Any


def plot_slope_range(
    data: List[float], x: List[str], parameter: Optional[str] = None, **kwargs: Any
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:

    title_infix = visualisation.aggregate.core.make_aggregate_title_infix(parameter)

    return visualisation.aggregate.core.plot_aggregate_values(
        data,
        "entropy_slope_median",
        x,
        title=f"Distribution of median slope across {title_infix} range",
        **kwargs,
    )
