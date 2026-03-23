import matplotlib.figure
import matplotlib.axes

import visualisation.core

from typing import Optional, Union, List, Tuple, Any


def plot_slope_dist(
    data: List[float], attribute: str, **kwargs: Any
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the slope distribution of a given parameter combination

    Args:
        data (List[float]): A list of values.
        attribute (str): The name of the series to model.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """

    return visualisation.core.plot_histogram(
        data,
        title=f"Slope distribution of {attribute} for chosen parameter combination",
        **kwargs,
    )
