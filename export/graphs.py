from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, List, Dict, Union, Tuple

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd

import visualisation.activation
import visualisation.base_rate
import visualisation.entropy
import visualisation.probabilities
import visualisation.multiplot

import export.runs


class GraphContext:
    EXPORT = 0
    DASHBOARD = 1


@dataclass
class GraphConfig:
    data_column: str  # What column does the required data come from?
    plot_func: Callable  # How can the figure be made?
    extra_args: Optional[Dict[str, Any]] = (
        None  # What extra arguments are needed to plot this figure?
    )
    is_mosaic: bool = False
    context: int = GraphContext.EXPORT


@dataclass
class MosaicConfig:
    layout: List[List[str]]  # Names of other graphs
    size: Tuple[int, int] = (10, 16)
    is_mosaic: bool = True
    context: int = GraphContext.DASHBOARD


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
    "composite_plot": MosaicConfig(
        layout=[
            ["ctx_activation_mean", "ctx_base_rate_mean"],
            ["ctx_entropy_mean", "ctx_probs_mean"],
        ],
        size=(12, 12),
    ),
}


def get_graph_names(context: int) -> List[str]:
    """Returns a list of the names of all available graphs

    Args:
        context (int): Context where the graphs will be used

    Returns:
        List[str]: A list of the names of all available graphs
    """

    return [
        graph_config
        for graph_config in list(graph_configs.keys())
        if graph_configs[graph_config].context == context
    ]


def get_graph_config(graph_name: str) -> Union[GraphConfig, MosaicConfig]:
    """Retrieve the configuration for a graph or mosaic graph

    Args:
        graph_name (str): Name of the graph

    Raises:
        ValueError: Raised if name of the graph does not reference an existing config

    Returns:
        Union[GraphConfig, MosaicConfig]: Configuration associated with the specified graph name
    """

    # First, retrieve the config for this graph (see above)
    if not graph_name in graph_configs:
        raise ValueError(f"'{graph_name}' is not a valid graph")

    return graph_configs[graph_name]


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
        config = get_graph_config(graph_name)

        # Check if mosaic plot
        if isinstance(config, MosaicConfig):
            # One by one, we replace the names of the graphs with the actual functions that build them
            plot_functions = []
            for row in config.layout:
                inner_functions = []
                for references_graph_name in row:
                    graph_function = generate_inner_lambda(data, references_graph_name)
                    inner_functions.append(graph_function)
                plot_functions.append(inner_functions)

            # Make the plot based on the functions
            figure = visualisation.multiplot.combine(plot_functions, config.size)
        else:
            # Make a single plot. We pass ax=None because there is no existing axis to hook into
            figure, ax = generate_inner_lambda(data, graph_name)(ax=None)

        graphs_output[graph_name] = figure

    return graphs_output


def generate_inner_lambda(data: Dict[str, Any], graph_name: str) -> Callable:
    """Generate the function which builds the graph specified by the graph name

    Args:
        data (Dict[str, Any]): Data dump of a specific parameter combination
        graph_name (str): Name of the graph to generate the function for

    Raises:
        TypeError: Raised if the graph name is associated with a mosaic function

    Returns:
        Callable: Function which generates the graph specified by the graph name
    """

    config = get_graph_config(graph_name)

    if isinstance(config, MosaicConfig):
        raise TypeError("Inner plot function cannot be of mosaic type")

    # Check if there are other arguments to be supplied, based on data argument
    kwargs = {}
    if config.extra_args:
        for arg_name, arg_func in config.extra_args.items():
            kwargs[arg_name] = arg_func(data)

        # Add common args
        kwargs["min_data"] = data[config.data_column]["min"]
        kwargs["max_data"] = data[config.data_column]["max"]

    # Make the plot function
    return lambda ax: config.plot_func(
        data[config.data_column]["mean"], **kwargs, ax=ax
    )
