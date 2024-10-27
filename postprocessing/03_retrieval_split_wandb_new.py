import os
import shelve
import numpy as np
import pandas as pd
from ast import literal_eval
import yaml
import wandb
from tqdm import tqdm

# Load configuration from conf.yaml
cap_col_num, run_num = "5x", 1
with open(f"./conf_yamls/cap_{cap_col_num}_conf.yaml", "rb") as stream:
    conf = yaml.full_load(stream)

# Extract WandB configuration
wandb_conf = conf.get("wandb_conf", {})

# Get the latest WandB run directory
api = wandb.Api()
runs = api.runs(wandb_conf["project"])  # Replace with your actual project name
latest_run = runs[len(runs) - run_num]
print(
    f"\nInitializing Run ID: {latest_run.id} && Run Name: {latest_run.name}\n"
)
wandb.init(project=wandb_conf["project"], id=latest_run.id)

# Construct the checkpoint directory path
ckp_fpath = os.path.join("./z_ckpts", latest_run.id)

# Extract data configuration
data_conf = conf["data_conf"]


def measure_single_item(qid, items, dataset_name, way):
    """
    Calculates retrieval metrics (recall@K, mAP@10) for a single query ID.

    Args:
        qid: Query ID.
        items: List of tuples (object ID, score, is_relevant) for the query.
        dataset_name: The name of the dataset being processed.
        way: The retrieval direction ("Audio2Txt" or "Txt2Audio").

    Returns:
        dict: Metrics for the single query.
    """
    objects = [i[0] for i in items]  # Extract object IDs
    scores = np.array([i[1] for i in items])  # Extract scores
    targets = np.array([i[2] for i in items])  # Extract relevance labels

    # Sort items by score in descending order
    desc_indices = np.argsort(scores, axis=-1)[::-1]
    targets = np.take_along_axis(arr=targets, indices=desc_indices, axis=-1)

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

    # Calculate mean average precision (mAP)
    positions = np.arange(1, 11, dtype=float)[targets[:10] > 0]
    if len(positions) > 0:
        precisions = np.divide(
            np.arange(1, len(positions) + 1, dtype=float), positions
        )
        avg_precision = np.sum(precisions, dtype=float) / np.sum(
            targets, dtype=float
        )
    else:
        avg_precision = 0.0

    return R1, R5, R10, avg_precision


def process_db(db_path, dataset_name, way):
    """
    Processes a shelve database and calculates retrieval metrics on the fly.

    Args:
        db_path: Path to the shelve database.
        dataset_name: The name of the dataset being processed.
        way: The retrieval direction ("Audio2Txt" or "Txt2Audio").

    Returns:
        dict: Average metrics for all queries.
    """
    total_R1, total_R5, total_R10, total_mAP = 0.0, 0.0, 0.0, 0.0
    count = 0

    with shelve.open(db_path) as stream:
        for qid, items in tqdm(
            stream.items(), desc=f"Processing {way} for {dataset_name}"
        ):
            R1, R5, R10, mAP = measure_single_item(
                qid, items, dataset_name, way
            )
            total_R1 += R1
            total_R5 += R5
            total_R10 += R10
            total_mAP += mAP
            count += 1

    # Calculate averages
    avg_R1 = total_R1 / count if count > 0 else 0.0
    avg_R5 = total_R5 / count if count > 0 else 0.0
    avg_R10 = total_R10 / count if count > 0 else 0.0
    avg_mAP = total_mAP / count if count > 0 else 0.0

    print(
        f"{dataset_name} {way} mAP: {avg_mAP:.3f}",
        f"R1: {avg_R1:.3f}",
        f"R5: {avg_R5:.3f}",
        f"R10: {avg_R10:.3f}",
        end="\n",
    )

    # Log the metrics to WandB
    wandb.log(
        {
            f"{dataset_name}_{way}_mAP": avg_mAP,
            f"{dataset_name}_{way}_R1": avg_R1,
            f"{dataset_name}_{way}_R5": avg_R5,
            f"{dataset_name}_{way}_R10": avg_R10,
        }
    )


if __name__ == "__main__":
    # Iterate through datasets
    for name in ["val_data", "eval_data"]:
        params = data_conf[name]
        name = name.replace("_data", "")

        # Load text data
        text_fpath = os.path.join(params["dataset"], params["text_data"])
        text_data = pd.read_csv(
            text_fpath, converters={"tokens": literal_eval}
        )
        print("\n\nLoaded", text_fpath)

        # Create mappings for text IDs, file IDs, and filenames
        tid2fid, tid2text, fid2fname = {}, {}, {}
        for idx in text_data.index:
            item = text_data.iloc[idx]
            tid2fid[item["tid"]] = item["fid"]
            tid2text[item["tid"]] = item["text"]
            fid2fname[item["fid"]] = item["fname"]

        # Process fid2items separately
        fid2items_db_fpath = os.path.join(
            ckp_fpath, f"{name}_fid_2_items_latest.db"
        )

        process_db(fid2items_db_fpath, dataset_name=name, way="Audio2Txt")

        # Process tid2items separately
        tid2items_db_fpath = os.path.join(
            ckp_fpath, f"{name}_tid_2_items_latest.db"
        )
        process_db(tid2items_db_fpath, dataset_name=name, way="Txt2Audio")

        # Output Text2Audio retrieval results
        results = []
        with shelve.open(tid2items_db_fpath) as tid_stream:
            for tid, items in tqdm(
                tid_stream.items(),
                desc=f"Processing Text2Audio results for {name}",
            ):
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
        print("Saved", result_fpath)

        # Log the results to WandB
        wandb.log(
            {f"retrieval_results_{name}": wandb.Table(dataframe=results_df)}
        )  # Log the dataframe as a table
