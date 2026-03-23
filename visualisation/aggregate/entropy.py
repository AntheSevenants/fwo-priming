import matplotlib.axes
import matplotlib.figure
import visualisation.entropy
import visualisation.aggregate.core

from typing import List, Optional, Union, Tuple, Any


def plot_entropy_range(
    data: List[float],
    x: List[str],
    num_constructions: int,
    parameter: Optional[str] = None,
    is_base_rate: bool = False,
    **kwargs: Any,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot an entropy range across parameter permutations.

    Args:
        data (List[float]): A list of outcome values for each parameter permutation.
        x (List[str]): A list of values for the x axis.
        num_constructions (int): The number of constructions in the simulation.
        parameter (Optional[str], optional): The name of the parameter that is being permutated. Defaults to None.
        is_base_rate (bool, optional): Whether the entropy measures are derived from the base rate. Defaults to False.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The finished graph
    """

    ylim = visualisation.entropy.ylim_from_num_constructions(num_constructions)
    title_infix = visualisation.aggregate.core.make_aggregate_title_infix(parameter)

    infix = "" if not is_base_rate else "base rate "

    return visualisation.aggregate.core.plot_aggregate_values(
        data,
        "entropy" if not is_base_rate else "base_rate_entropy",
        x,
        ylim=ylim,
        title=f"Distribution of mean {infix}entropy across {title_infix} range",
        **kwargs
    )
