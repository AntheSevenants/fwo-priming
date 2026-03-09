import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from typing import Callable


def combine_plots(
    ax1_func: Callable[[Axes], None],
    ax2_func: Callable[[Axes], None],
):
    """Combine two subplots on a single plot.

    Args:
        ax1_func (Callable[[Axes], None]): A lambda function with 'ax' as its only argument.
        ax2_func (Callable[[Axes], None]): A lambda function with 'ax' as its only argument.
    """

    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(16, 6))

    ax1_func(ax1)
    ax2_func(ax2)