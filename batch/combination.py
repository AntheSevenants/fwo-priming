from dataclasses import dataclass, field
from typing import Dict, List, Any, Callable, Tuple, Union

import numpy as np


@dataclass
class CombinationOperation:
    name: str
    operation: Callable[[np.ndarray], np.ndarray]


class CombinationOperations:
    Q1: CombinationOperation = CombinationOperation(
        name="q1", operation=lambda data_matrix: np.percentile(data_matrix, 25, axis=0)
    )
    Q3: CombinationOperation = CombinationOperation(
        name="q3", operation=lambda data_matrix: np.percentile(data_matrix, 75, axis=0)
    )
    MEDIAN: CombinationOperation = CombinationOperation(
        name="median", operation=lambda data_matrix: np.median(data_matrix, axis=0)
    )
    MIN: CombinationOperation = CombinationOperation(
        name="min", operation=lambda data_matrix: np.min(data_matrix, axis=0)
    )
    MAX: CombinationOperation = CombinationOperation(
        name="max", operation=lambda data_matrix: np.max(data_matrix, axis=0)
    )
    MEAN: CombinationOperation = CombinationOperation(
        name="mean", operation=lambda data_matrix: np.mean(data_matrix, axis=0)
    )


def get_combination_metrics(
    data_matrix: np.ndarray,
    operations: List[CombinationOperation]
) -> Dict[str, Union[List[np.float64], List[List[np.float64]]]]:
    combined_data_out = {}
    for operation in operations:
        combined_data_out[operation.name] = operation.operation(
            data_matrix
        ).tolist()

    return combined_data_out