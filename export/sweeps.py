import os
import pandas as pd

from typing import List

def get_sweeps(sweeps_dir: str) -> List[str]:
    """Returns a list of directory names associated with different model sweeps

    Args:
        sweeps_dir (str): The path to the directory where all sweeps are stored

    Returns:
        List[str]: List of directory names associated with model sweeps
    """

    sweep_dirs = next(os.walk(sweeps_dir))[1]
    sweep_dirs = sorted(sweep_dirs)

    return sweep_dirs

def get_run_infos(sweeps_dir: str, selected_sweep: str) -> pd.DataFrame:
    """Returns a dataframe containing the run info associated with different sweeps

    Args:
        sweeps_dir (str): The path to the directory where all sweeps are stored
        selected_sweep (str): The name of the sweep of interest

    Raises:
        FileNotFoundError: Raised if the run infos file cannot be found in the sweep directory

    Returns:
        pd.DataFrame: Dataframe containing the run info associated with different sweeps
    """

    selected_sweep_dir = make_selected_sweep_dir(sweeps_dir, selected_sweep)

    # I am not a French speaker, I just like using the word "infos" because it is goofy
    model_infos_path = os.path.join(selected_sweep_dir, "run_infos.csv")
    if not os.path.exists(model_infos_path):
        raise FileNotFoundError("Run infos CSV does nost exist")
    
    return pd.read_csv(model_infos_path)

def make_selected_sweep_dir(sweeps_dir: str, selected_sweep: str) -> str:
    """Make the path for the directory of the sweep of interest in the specified sweeps directory

    Args:
        sweeps_dir (str): The path to the directory where all sweeps are stored
        selected_sweep (str): The name of the sweep of interest

    Raises:
        FileNotFoundError: Raised if the selected sweep cannot be found in the sweep directory

    Returns:
        str: _description_
    """

    selected_sweep_dir = os.path.join(sweeps_dir, selected_sweep)
    if not os.path.exists(selected_sweep_dir):
        raise FileNotFoundError("Sweep directory does not exist")
    
    return selected_sweep_dir