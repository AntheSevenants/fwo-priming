from batch.runner import batch_run

from datetime import datetime

import pandas as pd
import numpy as np
import os
import argparse
import json
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
    br_df = br_df.sort_values(by=['run_id'])
    br_df.to_csv(csv_filename, index=False)

    # Post runs aggregation
    combinations = br_df.groupby("combination_id")["run_id"].agg(list).reset_index()
    # Go over each combination
    for index, row in combinations.iterrows():
        aggregated_data = { }
        aggregated_data_out = {}

        # Go over each run associated with this combination
        for run_id in row["run_id"]:
            # Load the run in memory
            run_dump_path = os.path.join(run_folder, f"{run_id}.json")
            with open(run_dump_path) as json_reader:
                json_content = json.loads(json_reader.read())

            for column_name in json_content.keys():
                if column_name.endswith("_mean"):
                    if column_name not in aggregated_data:
                        aggregated_data[column_name] = []

                    aggregated_data[column_name].append(
                        json_content[column_name]
                    )
            
        # Now aggregate, then turn into a list
        for column_name in aggregated_data:
            data_matrix = np.array(aggregated_data[column_name])
            aggregated_data_out[column_name] = {} 
            aggregated_data_out[column_name]["mean"] = data_matrix.mean(axis=0).tolist()
            aggregated_data_out[column_name]["min"] = data_matrix.min(axis=0).tolist()
            aggregated_data_out[column_name]["max"] = data_matrix.max(axis=0).tolist()
        
        aggregated_data["combination_id"] = row["combination_id"]

        combination_data_path = os.path.join(
            run_folder,
            f"combination_{row['combination_id']}.json"
        )

        with open(combination_data_path, "wt") as json_writer:
            json_writer.write(json.dumps(aggregated_data_out))
