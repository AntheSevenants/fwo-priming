from dataclasses import dataclass, field
from typing import Dict, List, Any, Callable, Tuple, Union

import numpy as np


@dataclass
class AggregateColumnOperation:
    name: str
    required_columns: List[str]
    operation: Callable[..., Union[np.ndarray, np.float64]]


class AggregateColumnOperations:
    MIN: AggregateColumnOperation = AggregateColumnOperation(
        name="min", required_columns=["mean"], operation=lambda data: np.min(data, axis=0)
    )
    MAX: AggregateColumnOperation = AggregateColumnOperation(
        name="max", required_columns=["mean"], operation=lambda data: np.max(data, axis=0)
    )
    MEAN: AggregateColumnOperation = AggregateColumnOperation(
        name="mean", required_columns=["mean"], operation=lambda data: np.mean(data, axis=0)
    )
    DELTA: AggregateColumnOperation = AggregateColumnOperation(
        name="delta",
        required_columns=["min", "max"],
        operation=lambda min_data, max_data: max_data.mean(axis=0) - min_data.mean(axis=0),
    )


@dataclass
class AggregateColumnConfig:
    data_column: str
    actions: List[AggregateColumnOperation] = field(
        default_factory=lambda: [
            AggregateColumnOperations.MIN,
            AggregateColumnOperations.MAX,
            AggregateColumnOperations.MEAN,
            AggregateColumnOperations.DELTA,
        ]
    )


aggregate_column_configs = {
    "entropy": AggregateColumnConfig(data_column="ctx_entropy_mean"),
    "activation": AggregateColumnConfig(data_column="ctx_activation_mean")
}


def get_aggregate_column_config(aggregate_column_name: str) -> AggregateColumnConfig:
    """Retrieve the configuration for an aggregate column

    Args:
        aggregate_column_name (str): Name of the aggregate column

    Raises:
        ValueError: Raised if the name of the aggregate column does not reference an existing config

    Returns:
        AggregateColumnConfig: Configuration associated with the specified aggregate column name
    """

    if not aggregate_column_name in aggregate_column_configs:
        raise ValueError(f"'{aggregate_column_name}' is not a valid aggregate column")

    return aggregate_column_configs[aggregate_column_name]


def get_aggregate_metrics(
    aggregated_data: Dict[str, Dict[str, List[Any]]],
    aggregate_column_names: List[str],
) -> Dict[str, Union[np.ndarray, np.float64]]:
    """Get aggregate metrics (min, max, mean) for specified properties of a parameter combination

    Args:
        aggregated_data (Dict[str, Dict[str, List[Any]]]): Aggregated per-combination data
        aggregate_column_names (List[str]): List of properties to get get data for

    Returns:
        Dict[str, Union[np.ndarray, np.float64]]: Dict with properties as keys and floats or numpy arrays as values
    """

    aggregate_metrics = {}

    for aggregate_column_name in aggregate_column_names:
        config = get_aggregate_column_config(aggregate_column_name)

        for action in config.actions:
            # This will be the name of the column in the output dataframe
            output_column_name = f"{aggregate_column_name}_{action.name}"
            # Automatically retrieve the columns that this specific action operation requires
            required_data = [
                np.array(aggregated_data[config.data_column][column])
                for column in action.required_columns
            ]
            aggregate_result = action.operation(*required_data)

            aggregate_metrics[output_column_name] = aggregate_result

    return aggregate_metrics