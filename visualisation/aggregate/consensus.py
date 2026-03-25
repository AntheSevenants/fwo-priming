import matplotlib.figure
import matplotlib.axes

import visualisation.core
import visualisation.aggregate.core

from typing import Optional, Union, List, Tuple, Any

def plot_consensus_aggregate(
    data: List[float], x: List[str], parameter: Optional[str] = None, **kwargs: Any
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the consensus distribution across parameter combinations

    Args:
        data (List[float]): Consensus share.
        x (List[str]): Parameter combination values
        str (Optional[str], optional): Name of the parameter that is being explored. Defaults to None

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """

    print(data)


    title_infix = visualisation.aggregate.core.make_aggregate_title_infix(parameter)

    return visualisation.core.plot_bar(
        data,
        x,
        ylim=[0, 1],
        title=f"Distribution of consensus across {title_infix} range",
        y_label="Percentage",
        **kwargs,
    )
