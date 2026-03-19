from dataclasses import dataclass, field
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
import visualisation.aggregate.entropy

import export.sweeps
import export.combinations

import batch.aggregate


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
    aggregate: bool = False
    is_mosaic: bool = False
    context: int = GraphContext.EXPORT


@dataclass
class MosaicConfig:
    layout: List[List[str]]  # Names of other graphs
    size: Tuple[int, int] = (10, 16)
    is_mosaic: bool = True
    context: int = GraphContext.DASHBOARD
    aggregate: bool = False


@dataclass
class AggregateSettings:
    combination_ids: List[int]
    parameter: str
    parameter_values: List[Any]

    def __init__(
        self,
        sweeps_dir: str,
        selected_sweep: str,
        combination_ids: List[int],
        parameter: str,
    ):
        self.combination_ids = combination_ids
        self.parameter = parameter

        run_infos = export.sweeps.get_run_infos(sweeps_dir, selected_sweep)
        self.parameter_values = sorted(
            run_infos[run_infos["combination_id"].isin(combination_ids)][parameter]
            .unique()
            .tolist()
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
    "ctx_base_rate_entropy_mean": GraphConfig(
        data_column="ctx_base_rate_entropy_mean",
        plot_func=visualisation.entropy.plot_ctx_entropy_mean,
        extra_args={
            "num_constructions": lambda data: len(data["ctx_base_rate_mean"]["mean"][0]),
            "base_rate": True
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
    "aggregate_entropy": GraphConfig(
        data_column="entropy",
        plot_func=visualisation.aggregate.entropy.plot_entropy_range,
        aggregate=True,
        context=GraphContext.DASHBOARD,
        extra_args={
            "num_constructions": lambda data: len(data.iloc[0]["activation_mean"])
        },
    ),
    "aggregate_base_rate_entropy": GraphConfig(
        data_column="base_rate_entropy",
        plot_func=visualisation.aggregate.entropy.plot_entropy_range,
        aggregate=True,
        context=GraphContext.DASHBOARD,
        extra_args={
            "num_constructions": lambda data: len(data.iloc[0]["activation_mean"]),
            "base_rate": True
        },
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
        and not graph_configs[graph_config].aggregate
    ]


def get_aggregate_graph_names(context: int) -> List[str]:
    """Returns a list of the names of all available aggregate graphs

    Args:
        context (int): Context where the graphs will be used

    Returns:
        List[str]: A list of the names of all available graphs
    """

    return [
        graph_config
        for graph_config in list(graph_configs.keys())
        if graph_configs[graph_config].context == context
        and graph_configs[graph_config].aggregate
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
    combination_ids: Union[int, List[int]],
    graphs: List[str],
    aggregate: Optional[AggregateSettings] = None,
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
        ValueError: Raised if multiple combination IDs appear without an aggregate configuration

    Returns:
        Dict[str, matplotlib.figure.Figure]: Dictionary with graph names as keys and generated graphs as values
    """

    # Now, we can build the desired graphs and save them
    graphs_output = {}

    # If only a single combination_id is given, this is a single graph
    if isinstance(combination_ids, int) and aggregate is None:
        # Retrieve the data for the single combination
        combination_id = combination_ids
        data = export.combinations.get_combination_data(
            sweeps_dir, selected_sweep, combination_id
        )
    elif isinstance(combination_ids, list) and aggregate is not None:
        # Get the combination infos dataframe
        combination_infos = export.sweeps.get_combination_infos(
            sweeps_dir, selected_sweep
        )
        # Filter for the required combinations
        data = combination_infos[
            combination_infos["combination_id"].isin(combination_ids)
        ]
    else:
        raise ValueError(
            "Unrecognised combination of combination IDs and aggregate settings"
        )

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
            figure, ax = generate_inner_lambda(
                data, graph_name, aggregate_config=aggregate
            )(ax=None)

        graphs_output[graph_name] = figure

    return graphs_output


def generate_inner_lambda(
    data: Union[Dict[str, Any], pd.DataFrame], graph_name: str, aggregate_config=None
) -> Callable:
    """Generate the function which builds the graph specified by the graph name

    Args:
        data (Union[Dict[str, Any], pd.DataFrame]): Data dump of a specific parameter combination, or combinations
        graph_name (str): Name of the graph to generate the function for
        aggregate_config (AggregateSettings): Configuration for aggregate graphs

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
            # extra_arg is a lambda function
            if isinstance(arg_func, Callable):
                kwargs[arg_name] = arg_func(data)
            # extra_arg is a constant
            else:
                kwargs[arg_name] = arg_func

    # Regular graph
    if aggregate_config is None:
        # Add common args
        kwargs["min_data"] = data[config.data_column]["min"]
        kwargs["max_data"] = data[config.data_column]["max"]

        # Make the plot function
        return lambda ax: config.plot_func(
            data[config.data_column]["mean"], **kwargs, ax=ax
        )
    else:
        kwargs["min_data"] = data[
            batch.aggregate.make_aggregate_output_name(config.data_column, "min")
        ]
        kwargs["max_data"] = data[
            batch.aggregate.make_aggregate_output_name(config.data_column, "max")
        ]

        return lambda ax: config.plot_func(
            data[
                batch.aggregate.make_aggregate_output_name(config.data_column, "mean")
            ],
            aggregate_config.parameter_values,
            parameter=aggregate_config.parameter,
            **kwargs,
            ax=ax,
        )
