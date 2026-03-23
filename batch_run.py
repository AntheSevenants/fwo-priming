from batch.runner import batch_run

from datetime import datetime
from tqdm.auto import tqdm

import pandas as pd
import numpy as np
import os
import argparse
import json
import random
import math

import batch.profiles
import batch.aggregate
import batch.combination
import export.sweeps
import export.runs
import export.combinations

from model.model import PrimingModel

parser = argparse.ArgumentParser(description="batch_run - volt go brr again")
parser.add_argument("profile", type=str, help="regular")
parser.add_argument(
    "iterations", type=int, default=1, help="number of iterations, default = 1"
)
args = parser.parse_args()

NUM_STEPS = 10000
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
        max_steps=NUM_STEPS,
        number_processes=None,
        data_collection_period=100,
        display_progress=True,
    )

    csv_filename = export.sweeps.make_run_infos_path(SWEEPS_DIR, current_sweep)
    br_df = pd.DataFrame(results)
    br_df = br_df.sort_values(by=['run_id'])
    br_df.to_csv(csv_filename, index=False)

    # Post runs aggregation
    combinations = br_df.groupby("combination_id")["run_id"].agg(list).reset_index()
    combination_infos_df_rows = []
    # Go over each combination
    for index, row in tqdm(
        combinations.iterrows(),
        total=len(combinations),
        desc="Computing combination metrics"
        ):
        aggregated_data = { }
        aggregated_data_out = {}

        # Go over each run associated with this combination
        for run_id in row["run_id"]:
            # Load the run in memory
            json_content = export.runs.load_dataframe(SWEEPS_DIR, current_sweep, run_id)

            for column_name in json_content.keys():
                if column_name.endswith("_mean") or column_name.endswith("_median"):
                    if column_name not in aggregated_data:
                        aggregated_data[column_name] = []

                    aggregated_data[column_name].append(
                        json_content[column_name]
                    )
            
        # Now aggregate, then turn into a list
        for column_name in aggregated_data:
            data_matrix = np.array(aggregated_data[column_name])
            aggregated_data_out[column_name] = batch.combination.get_combination_metrics(
                data_matrix,
                [batch.combination.CombinationOperations.Q1,
                 batch.combination.CombinationOperations.Q3,
                 batch.combination.CombinationOperations.MEDIAN,
                 batch.combination.CombinationOperations.MEAN,
                 batch.combination.CombinationOperations.MIN,
                 batch.combination.CombinationOperations.MAX]
            )
        
        aggregated_data["combination_id"] = row["combination_id"]

        combination_data_path = export.combinations.make_combination_data_path(
            SWEEPS_DIR,
            current_sweep,
            row["combination_id"]
        )

        with open(combination_data_path, "wt") as json_writer:
            json_writer.write(json.dumps(aggregated_data_out))

        # Add row to combination_infos records
        combination_aggregate_metrics = batch.aggregate.get_aggregate_metrics(
            aggregated_data_out,
            list(batch.aggregate.aggregate_column_configs.keys())
        )
        combination_aggregate_metrics = { 
            "combination_id": row["combination_id"], 
            **combination_aggregate_metrics
        }
        combination_infos_df_rows.append(combination_aggregate_metrics)

    # Write combination_infos file
    combination_infos_path = export.sweeps.make_combination_infos_path(
        SWEEPS_DIR, current_sweep
    )
    combination_infos_df = pd.DataFrame.from_records(combination_infos_df_rows)
    combination_infos_df.to_json(
        combination_infos_path,
        orient='records'
    )