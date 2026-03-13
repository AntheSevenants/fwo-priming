from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, List, Dict, Union

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd

import visualisation.activation
import visualisation.base_rate
import visualisation.entropy
import visualisation.probabilities

import export.runs


@dataclass
class GraphConfig:
    data_column: str  # What column does the required data come from?
    plot_func: Callable  # How can the figure be made?
    extra_args: Optional[Dict[str, Any]] = (
        None  # What extra arguments are needed to plot this figure?
    )


graph_configs = {
    "ctx_activation_mean": GraphConfig(
        data_column="ctx_activation_mean",
        plot_func=visualisation.activation.plot_ctx_activation_mean,
    ),
    "ctx_base_rate_mean": GraphConfig(
        data_column="ctx_base_rate_mean",
        plot_func=visualisation.base_rate.plot_ctx_base_rate_mean,
    ),
    "ctx_entropy_mean": GraphConfig(
        data_column="ctx_entropy_mean",
        plot_func=visualisation.entropy.plot_ctx_entropy_mean,
        extra_args={
            "num_constructions": lambda data: len(data["ctx_base_rate_mean"]["mean"][0])
        },
    ),
    "ctx_probs_mean": GraphConfig(
        data_column="ctx_probs_mean",
        plot_func=visualisation.probabilities.plot_ctx_probs_mean,
    ),
}


def get_graph_names() -> List[str]:
    """Returns a list of the names of all available graphs

    Returns:
        List[str]: A list of the names of all available graphs
    """

    return list(graph_configs.keys())


def generate_graphs(
    sweeps_dir: str,
    selected_sweep: str,
    combination_id: int,
    graphs: List[str],
    disable_title=False,
) -> Dict[str, matplotlib.figure.Figure]:
    """Generate the specified graphs depending on the given sweep

    Args:
        sweeps_dir (str): Path to the directory where all sweeps are stored
        selected_sweep (str): Name of the sweep of interest
        combination_id (int): ID of the unique parameter combination
        graphs (List[str]): List of names of the graphs to be generated
        disable_title (bool, optional): Whether to show a title for this graph. Defaults to False.

    Raises:
        ValueError: Raised if a supplied graph name does not have an associated graph

    Returns:
        Dict[str, matplotlib.figure.Figure]: Dictionary with graph names as keys and generated graphs as values
    """

    # Now, we can build the desired graphs and save them
    graphs_output = {}

    # Retrieve the data for this combination
    data = export.runs.get_combination_data(sweeps_dir, selected_sweep, combination_id)

    # We go over all requested graphs and generate them
    for graph_name in graphs:
        # First, retrieve the config for this graph (see above)
        if not graph_name in graph_configs:
            raise ValueError(f"'{graph_name}' is not a valid graph")
        config = graph_configs[graph_name]

        # Check if there are other arguments to be supplied, based on data argument
        kwargs = {}
        if config.extra_args:
            for arg_name, arg_func in config.extra_args.items():
                kwargs[arg_name] = arg_func(data)

        # Add common args
        kwargs["min_data"] = data[config.data_column]["min"]
        kwargs["max_data"] = data[config.data_column]["max"]

        # Make the actual plot
        figure = config.plot_func(data[config.data_column]["mean"], **kwargs)

        graphs_output[graph_name] = figure

    return graphs_output
