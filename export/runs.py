import os
import json
import pandas as pd
import numpy as np

import export.sweeps

from typing import Dict, Any


def make_run_data_path(sweeps_dir: str, selected_sweep: str, run_id: int) -> str:
    """Make the path for where run data JSON is stored

    Args:
        sweeps_dir (str): The path to the directory where all sweeps are stored
        selected_sweep (str): The name of the sweep of interest
        run_id (int): ID of the run of interest

    Returns:
        str: Path where run data is stored
    """

    sweep_dir = export.sweeps.make_selected_sweep_dir(sweeps_dir, selected_sweep)
    return os.path.join(sweep_dir, f"{run_id}.json")


def make_combination_data_path(sweeps_dir: str, selected_sweep: str, combination_id: int) -> str:
    """Make the path for where combnation data JSON is stored

    Args:
        sweeps_dir (str): The path to the directory where all sweeps are stored
        selected_sweep (str): The name of the sweep of interest
        combination_id (int): Unique ID for the parameter selection

    Returns:
        str: Path where combination data is stored
    """

    sweep_dir = export.sweeps.make_selected_sweep_dir(sweeps_dir, selected_sweep)
    return os.path.join(sweep_dir, f"combination_{combination_id}.json")


def load_dataframe(sweeps_dir: str, selected_sweep: str, run_id: int) -> Dict[str, Any]:
    """Load a data dump for a specific run

    Args:
        sweeps_dir (str): The path to the directory where all sweeps are stored
        selected_sweep (str): The name of the sweep of interest
        run_id (int): interest
        combination_id (int): Unique

    Returns:
        Dict[str, Any]: Unserialised data dump of the specified run
    """

    sweep_dir = export.sweeps.make_selected_sweep_dir(sweeps_dir, selected_sweep)
    run_data_path = os.path.join(sweep_dir, f"{run_id}.json")

    # Load the selected simulation run from disk
    with open(run_data_path, "rt") as run_file:
        # Load the dataframe-as-json from disk
        # There is no need to really turn it into a dataframe, downstream will figure it out
        json_object = json.loads(run_file.read())

        return json_object


def get_combination_data(
    sweeps_dir: str, selected_sweep: str, combination_id: int
) -> Dict[str, Any]:
    """Load a data dump for a specific combination

    Args:
        sweeps_dir (str): The path to the directory where all sweeps are stored
        selected_sweep (str): The name of the sweep of interest
        combination_id (int): Unique ID for the parameter selection

    Returns:
        Dict[str, Any]: Unserialised data dump of the specified combination
    """

    combination_data_path = make_combination_data_path(sweeps_dir, selected_sweep, combination_id)

    with open(combination_data_path, "rt") as reader:
        data = json.loads(reader.read())

    return data
