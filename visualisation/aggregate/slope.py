import matplotlib.figure
import matplotlib.axes

import visualisation.aggregate.core

from typing import Optional, Union, List, Tuple, Any


def plot_slope_range(
    data: List[float], x: List[str], parameter: Optional[str] = None, **kwargs: Any
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot a range of slope values across parameter permutations.

    Args:
        data (List[float]): A list of outcome values (slopes) for each parameter permutation.
        x (List[str]): A list of values for the x axis.
        parameter (Optional[str], optional): The name of the parameter that is being permutated. Defaults to None.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """

    title_infix = visualisation.aggregate.core.make_aggregate_title_infix(parameter)

    return visualisation.aggregate.core.plot_aggregate_values(
        data,
        x,
        title=f"Distribution of median slope across {title_infix} range",
        **kwargs,
    )
