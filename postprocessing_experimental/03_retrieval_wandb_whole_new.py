import os
import sqlite3
import numpy as np
import pandas as pd
from ast import literal_eval
import yaml
import wandb
from tqdm import tqdm

# Load configuration from conf.yaml
cap_col_num, run_num = 1, 5
with open(f"./conf_yamls/exp_{cap_col_num}x_conf.yaml", "rb") as stream:
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
# wandb.init(project=wandb_conf['project'], id=latest_run.id)

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


def process_db_on_the_fly(db_path, dataset_name):
    """
    Processes a single SQLite database and calculates retrieval metrics on the fly for both Audio2Text and Text2Audio.

    Args:
        db_path: Path to the SQLite database.
        dataset_name: The name of the dataset being processed.

    Returns:
        None. Prints metrics and saves results.
    """
    # Initialize totals for Audio2Text and Text2Audio
    fid2_total_R1, fid2_total_R5, fid2_total_R10, fid2_total_mAP = (
        0.0,
        0.0,
        0.0,
        0.0,
    )
    tid2_total_R1, tid2_total_R5, tid2_total_R10, tid2_total_mAP = (
        0.0,
        0.0,
        0.0,
        0.0,
    )
    fid2_count, tid2_count = 0, 0

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Process each row in the database
    cursor.execute(
        "SELECT fid, tid, score FROM cross_modal_scores"
    )  # Adjust table/column names if needed
    fid2items, tid2items = {}, {}

    # Retrieve each row from the database in a loop
    for fid, tid, score in tqdm(
        cursor.fetchall(), desc=f"Processing {dataset_name}"
    ):
        is_relevant = tid2fid[tid] == fid

        # Group items by fid for Audio2Text
        fid2items.setdefault(fid, []).append((tid, score, is_relevant))

        # Group items by tid for Text2Audio
        tid2items.setdefault(tid, []).append((fid, score, is_relevant))

    # Measure Audio2Text retrieval
    for fid, items in fid2items.items():
        R1, R5, R10, mAP = measure_single_item(
            fid, items, dataset_name, "Audio2Txt"
        )
        fid2_total_R1 += R1
        fid2_total_R5 += R5
        fid2_total_R10 += R10
        fid2_total_mAP += mAP
        fid2_count += 1

    # Measure Text2Audio retrieval
    for tid, items in tid2items.items():
        R1, R5, R10, mAP = measure_single_item(
            tid, items, dataset_name, "Txt2Audio"
        )
        tid2_total_R1 += R1
        tid2_total_R5 += R5
        tid2_total_R10 += R10
        tid2_total_mAP += mAP
        tid2_count += 1

    # Calculate and print average metrics for Audio2Text
    avg_fid2_R1 = fid2_total_R1 / fid2_count if fid2_count > 0 else 0.0
    avg_fid2_R5 = fid2_total_R5 / fid2_count if fid2_count > 0 else 0.0
    avg_fid2_R10 = fid2_total_R10 / fid2_count if fid2_count > 0 else 0.0
    avg_fid2_mAP = fid2_total_mAP / fid2_count if fid2_count > 0 else 0.0
    print(
        f"{dataset_name} Audio2Txt mAP: {avg_fid2_mAP:.3f}",
        f"R1: {avg_fid2_R1:.3f}",
        f"R5: {avg_fid2_R5:.3f}",
        f"R10: {avg_fid2_R10:.3f}",
    )

    # Calculate and print average metrics for Text2Audio
    avg_tid2_R1 = tid2_total_R1 / tid2_count if tid2_count > 0 else 0.0
    avg_tid2_R5 = tid2_total_R5 / tid2_count if tid2_count > 0 else 0.0
    avg_tid2_R10 = tid2_total_R10 / tid2_count if tid2_count > 0 else 0.0
    avg_tid2_mAP = tid2_total_mAP / tid2_count if tid2_count > 0 else 0.0
    print(
        f"{dataset_name} Txt2Audio mAP: {avg_tid2_mAP:.3f}",
        f"R1: {avg_tid2_R1:.3f}",
        f"R5: {avg_tid2_R5:.3f}",
        f"R10: {avg_tid2_R10:.3f}",
    )

    # Close the database connection
    conn.close()


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
        print("\n\nLoaded", text_fpath)

        # Create mappings for text IDs, file IDs, and filenames
        tid2fid, tid2text, fid2fname = {}, {}, {}
        for idx in text_data.index:
            item = text_data.iloc[idx]
            tid2fid[item["tid"]] = item["fid"]
            tid2text[item["tid"]] = item["text"]
            fid2fname[item["fid"]] = item["fname"]

        # Process the single SQLite database
        db_fpath = os.path.join(ckp_fpath, f"{name}_xmodal_scores.db")
        process_db_on_the_fly(db_fpath, dataset_name=name)
