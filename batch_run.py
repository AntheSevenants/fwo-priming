from batch.runner import batch_run

from datetime import datetime

import pandas as pd
import argparse
import dataclasses
import json

import batch.profiles
import batch.sweep_info
import export.sweeps
import batch_run_post

from model.model import PrimingModel

parser = argparse.ArgumentParser(description="batch_run - volt go brr again")
parser.add_argument("profile", type=str, help="regular")
parser.add_argument(
    "iterations", type=int, default=1, help="number of iterations, default = 1"
)
args = parser.parse_args()

sweep_info = batch.sweep_info.SweepInfo(
    num_steps=50000,
    datacollector_step_ratio=0.01
)

SWEEPS_DIR = "sweeps/"

if not args.profile in batch.profiles.params:
    raise ValueError("Unrecognised profile")

params = batch.profiles.params[args.profile]

if __name__ == "__main__":
    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Name of the current sweep
    current_sweep = f"{args.profile}-{date_time}"

    results = batch_run(
        PrimingModel,
        SWEEPS_DIR,
        current_sweep,
        parameters=params,
        iterations=args.iterations,
        max_steps=sweep_info.num_steps,
        datacollector_step_size=sweep_info.datacollector_step_size,
        number_processes=None,
        data_collection_period=100,
        display_progress=True,
    )

    csv_filename = export.sweeps.make_run_infos_path(SWEEPS_DIR, current_sweep)
    br_df = pd.DataFrame(results)
    br_df = br_df.sort_values(by=['run_id'])
    br_df.to_csv(csv_filename, index=False)
    
    # Write meta information about this parameter sweep
    sweep_info_path = export.sweeps.make_sweep_info_path(SWEEPS_DIR, current_sweep)
    with open(sweep_info_path, "wt") as writer:
        sweep_info_dict = dataclasses.asdict(sweep_info)
        writer.write(json.dumps(sweep_info_dict))

    batch_run_post.do_post_run_aggregation(current_sweep)