import matplotlib.pyplot as plt

from typing import Callable


def combine_plots(
    ax1_func: Callable[[plt.Axes], None],
    ax2_func: Callable[[plt.Axes], None],
):
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(16, 6))

    ax1_func(ax1)
    ax2_func(ax2)