import dbm
import json
import os
import shelve
from ast import literal_eval
from dbm import dumb

import numpy as np
import pandas as pd
import yaml
import wandb

dbm._defaultmod = dumb
dbm._modules = {"dbm.dumb": dumb}

# Load configuration from conf.yaml
cap_col_num, run_num = 2, 1
with open(f"./conf_yamls/exp_{cap_col_num}x_conf.yaml", "rb") as stream:
    conf = yaml.full_load(stream)

# Extract WandB configuration
wandb_conf = conf.get("wandb_conf", {})

# Get the latest WandB run directory
api = wandb.Api()
runs = api.runs(wandb_conf["project"])  # Replace with your actual project name
latest_run = runs[len(runs) - run_num]

# Construct the checkpoint directory path
ckp_fpath = os.path.join("./z_ckpts", latest_run.id)

# Extract data configuration
data_conf = conf["data_conf"]

# Initialize WandB run using the latest_run
# wandb.init(project=wandb_conf['project'], id=latest_run.id)


def measure(qid2items, dataset_name, way):
    """
    Calculates retrieval metrics (recall@K, mAP@10) over sample queries.

    Args:
        qid2items: Dictionary where keys are query IDs and values are lists of tuples (object ID, score, is_relevant).
        dataset_name: The name of the dataset being processed (e.g., "train", "val", "eval").

    Returns:
        None (prints the metrics to the console and logs them to WandB).
    """
    qid_R1s, qid_R5s, qid_R10s, qid_mAP10s = [], [], [], []

    for qid, items in qid2items.items():
        objects = [i[0] for i in items]  # Extract object IDs
        scores = np.array([i[1] for i in items])  # Extract scores
        targets = np.array([i[2] for i in items])  # Extract relevance labels

        # Sort items by score in descending order
        desc_indices = np.argsort(scores, axis=-1)[::-1]
        targets = np.take_along_axis(
            arr=targets, indices=desc_indices, axis=-1
        )
        # targets = targets[desc_indices]

        # Calculate recall at cutoffs 1, 5, and 10
        R1 = (
            np.sum(targets[:1], dtype=float) / np.sum(targets, dtype=float)
            if np.sum(targets) > 0
            else 0.0
        )
        R5 = (
            np.sum(targets[:5], dtype=float) / np.sum(targets, dtype=float)
            if np.sum(targets) > 0
            else 0.0
        )
        R10 = (
            np.sum(targets[:10], dtype=float) / np.sum(targets, dtype=float)
            if np.sum(targets) > 0
            else 0.0
        )

        qid_R1s.append(R1)
        qid_R5s.append(R5)
        qid_R10s.append(R10)

        # Calculate mean average precision (mAP)
        positions = np.arange(1, 11, dtype=float)[targets[:10] > 0]
        if len(positions) > 0:
            precisions = np.divide(
                np.arange(1, len(positions) + 1, dtype=float), positions
            )
            avg_precision = np.sum(precisions, dtype=float) / np.sum(
                targets, dtype=float
            )
            qid_mAP10s.append(avg_precision)
        else:
            qid_mAP10s.append(0.0)

    # Print the average metrics
    print(
        f"{dataset_name} mAP: {np.mean(qid_mAP10s):.3f}",
        f"R1: {np.mean(qid_R1s):.3f}",
        f"R5: {np.mean(qid_R5s):.3f}",
        f"R10: {np.mean(qid_R10s):.3f}",
        end="\n",
    )

    # Log the metrics to WandB with dataset name as a prefix
    # wandb.log({
    #     f"{dataset_name}_{way}_mAP": np.mean(qid_mAP10s),
    #     f"{dataset_name}_{way}_R1": np.mean(qid_R1s),
    #     f"{dataset_name}_{way}_R5": np.mean(qid_R5s),
    #     f"{dataset_name}_{way}_R10": np.mean(qid_R10s)
    # })


if __name__ == "__main__":
    # Iterate through datasets
    for name in data_conf.keys():
        params = data_conf[name]
        name = name.replace("_data", "")

        # Load text data
        text_fpath = os.path.join(params["dataset"], params["text_data"])
        text_data = pd.read_csv(
            text_fpath, converters={"tokens": literal_eval}
        )
        print("\n\n Load", text_fpath, end="\n")

        # Create mappings for text IDs, file IDs, and filenames
        tid2fid, tid2text, fid2fname = {}, {}, {}
        for idx in text_data.index:
            item = text_data.iloc[idx]
            tid2fid[item["tid"]] = item["fid"]
            tid2text[item["tid"]] = item["text"]
            fid2fname[item["fid"]] = item["fname"]

        # Load cross-modal scores from the shelve file
        score_fpath = os.path.join(ckp_fpath, f"{name}_xmodal_scores.db")
        with shelve.open(score_fpath) as stream:
            fid2items, tid2items = {}, {}

            # Iterate through the audio file IDs in the shelve file
            for fid, group_scores in stream.items():

                # Audio2Text retrieval: Create a list of tuples (tid, score, is_relevant) for each audio file
                fid2items[fid] = [
                    (tid, group_scores[tid], tid2fid[tid] == fid)
                    for tid in group_scores
                ]

                # Text2Audio retrieval: Create a list of tuples (fid, score, is_relevant) for each text ID
                for tid, score in group_scores.items():
                    if tid not in tid2items:
                        tid2items[tid] = [(fid, score, tid2fid[tid] == fid)]
                    else:
                        tid2items[tid].append(
                            (fid, score, tid2fid[tid] == fid)
                        )
                # print(f"For {fid}: len(fid2items) = {len(fid2items[fid])} and len(tid2items) = {len(tid2items[fid])}")
            # Ensure there is data to measure
            if fid2items:
                print(f"Audio2Text retrieval for {name}")
                measure(
                    fid2items, dataset_name=name, way="Audio2Txt"
                )  # Pass dataset name
            if tid2items:
                print(f"Text2Audio retrieval for {name}")
                measure(
                    tid2items, dataset_name=name, way="Txt2Audio"
                )  # Pass dataset name

            # Output Text2Audio retrieval results
            results = []
            for tid, items in tid2items.items():
                objects = [i[0] for i in items]  # Extract file IDs
                scores = np.array([i[1] for i in items])  # Extract scores

                # Sort items by score in descending order
                desc_indices = np.argsort(scores, axis=-1)[::-1]

                # Create a list of text and top-10 retrieved audio filenames
                line = [tid2text[tid]] + [
                    fid2fname[objects[idx]] for idx in desc_indices[:10]
                ]
                results.append(line)

            # Create a pandas DataFrame and save the results to a CSV file
            results_df = pd.DataFrame(data=results)
            result_fpath = os.path.join(ckp_fpath, f"{name}.t2a_retrieval.csv")
            results_df.to_csv(result_fpath, index=False)
            print("Save", result_fpath)

            # Log the results to WandB
            # wandb.log({f"retrieval_results_{name}": wandb.Table(dataframe=results_df)})  # Log the dataframe as a table
