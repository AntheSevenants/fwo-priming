import matplotlib.figure
import matplotlib.axes

import visualisation.core

from typing import Optional, Union, List, Tuple, Any


def plot_consensus_dist(
    data: List[float], **kwargs: Any
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the consensus distribution of a given parameter combination

    Args:
        data (List[float]): Consensus share. In a list, but I actually only expect one item.
        attribute (str): The name of the series to model.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """

    if len(data) > 1:
        raise ValueError("Data length maximum for consensus graph is 1")

    _data = data[0]

    y_values = [ _data, 1 - _data ]
    x_values = [ "Consensus reached", "No consensus reached" ]

    return visualisation.core.plot_bar(
        y_values,
        x_values,
        title=f"Distribution of consensus for this parameter combination",
        y_label="Percentage",
        **kwargs,
    )
