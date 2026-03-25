import matplotlib.figure
import matplotlib.axes

import visualisation.core

from typing import Optional, Union, List, Tuple, Any


def plot_consensus_dist(
    data: float, **kwargs: Any
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the consensus distribution of a given parameter combination

    Args:
        data (float): Consensus share.
        attribute (str): The name of the series to model.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """

    y_values = [ data, 1 - data ]
    x_values = [ "Consensus reached", "No consensus reached" ]

    return visualisation.core.plot_bar(
        y_values,
        x_values,
        title=f"Distribution of consensus for this parameter combination",
        y_label="Percentage",
        **kwargs,
    )
