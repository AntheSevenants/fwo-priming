import os
import json
import pandas as pd
import numpy as np

from typing import Dict, Any


def load_dataframe(run_path: str) -> Dict[str, Any]:
    """Load a data dump for a specific run given the data dump path. Currently unused.

    Args:
        run_path (str): Path to the JSON data dump

    Returns:
        Dict[str, Any]: Unserialised data dump
    """

    # Load the selected simulation run from disk
    with open(run_path, "rt") as run_file:
        # Load the dataframe-as-json from disk
        # There is no need to really turn it into a dataframe, downstream will figure it out
        json_object = json.loads(run_file.read())

        return json_object


def get_combination_data(sweeps_dir: str, selected_sweep: str, combination_id: int) -> Dict[str, Any]:
    """Load a data dump for a specific combination

    Args:
        sweeps_dir (str): The path to the directory where all sweeps are stored
        selected_sweep (str): The name of the sweep of interest
        combination_id (int): Unique ID for the parameter selection

    Returns:
        Dict[str, Any]: Unserialised data dump of the specified combination
    """

    combination_data_path = os.path.join(
        sweeps_dir, selected_sweep, f"combination_{combination_id}.json"
    )

    with open(combination_data_path, "rt") as reader:
        data = json.loads(reader.read())

    return data
