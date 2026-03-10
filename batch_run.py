from batch.runner import batch_run

from datetime import datetime

import pandas as pd
import numpy as np
import os
import argparse
import random
import math

import batch.profiles

from model.model import PrimingModel

parser = argparse.ArgumentParser(description="batch_run - volt go brr again")
parser.add_argument("profile", type=str, help="regular")
parser.add_argument(
    "iterations", type=int, default=1, help="number of iterations, default = 1"
)
args = parser.parse_args()

NUM_STEPS = 10000

if not args.profile in batch.profiles.params:
    raise ValueError("Unrecognised profile")

params = batch.profiles.params[args.profile]

if __name__ == "__main__":
    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    run_folder = f"sweeps/{args.profile}-{date_time}/"
    os.makedirs(run_folder, exist_ok=True)

    results = batch_run(
        PrimingModel,
        run_folder,
        parameters=params,
        iterations=args.iterations,
        max_steps=NUM_STEPS,
        number_processes=None,
        data_collection_period=100,
        display_progress=True,
    )

    csv_filename = f"{run_folder}run_infos.csv"
    br_df = pd.DataFrame(results)
    br_df.to_csv(csv_filename, index=False)
