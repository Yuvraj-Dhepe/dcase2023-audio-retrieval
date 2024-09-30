import os
import shelve
import pandas as pd
from tqdm import tqdm
from ast import literal_eval
import dbm
from dbm import dumb
import wandb
import yaml

dbm._defaultmod = dumb
dbm._modules = {"dbm.dumb": dumb}

# Load configuration from conf.yaml
cap_col_num, run_num = '5x', 1

with open(f"./conf_yamls/cap_{cap_col_num}_conf.yaml", "rb") as stream:
    conf = yaml.full_load(stream)

# Extract WandB configuration
wandb_conf = conf.get("wandb_conf", {})

# Get the latest WandB run directory
api = wandb.Api()
runs = api.runs(wandb_conf['project'])  # Replace with your actual project name
latest_run = runs[len(runs) - run_num]

# Construct the checkpoint directory path
print(f"\nInitializing Run ID: {latest_run.id} && Run Name: {latest_run.name}\n")

ckp_fpath = os.path.join("./z_ckpts", latest_run.id)

# Extract data configuration
data_conf = conf["data_conf"]

# Iterate through datasets
for name in ['val_data','eval_data']:
    params = data_conf[name]
    name = name.replace("_data", "")

    # Load text data
    text_fpath = os.path.join(params["dataset"], params["text_data"])
    text_data = pd.read_csv(text_fpath, converters={"tokens": literal_eval})
    print("\n\nLoaded", text_fpath)

    # Create mappings for text IDs, file IDs, and filenames
    tid2fid, tid2text, fid2fname = {}, {}, {}
    for idx in text_data.index:
        item = text_data.iloc[idx]
        tid2fid[item["tid"]] = item["fid"]
        tid2text[item["tid"]] = item["text"]
        fid2fname[item["fid"]] = item["fname"]

    # Load cross-modal scores from the shelve file
    score_fpath = os.path.join(ckp_fpath, f"{name}_xmodal_scores.db")

    # Initialize shelve databases for storage
    fid2items_db_fpath = os.path.join(ckp_fpath, f"{name}_fid_2_items_latest.db")
    tid2items_db_fpath = os.path.join(ckp_fpath, f"{name}_tid_2_items_latest.db")

    with shelve.open(score_fpath) as stream, shelve.open(fid2items_db_fpath, 'c') as fid_stream, shelve.open(tid2items_db_fpath, 'c') as tid_stream:

        # Create an empty dictionary to store results for tid2items
        all_tid2items = {}

        # Iterate through the audio file IDs in the shelve file
        for fid, group_scores in tqdm(stream.items(), desc="Processing Items"):

            # Audio2Text retrieval: Create a list of tuples (tid, score, is_relevant) for each audio file
            fid_stream[fid] = [(tid, group_scores[tid], tid2fid[tid] == fid) for tid in group_scores]

            # Text2Audio retrieval: Create a list of tuples (fid, score, is_relevant) for each text ID
            for tid, score in group_scores.items():
                if tid not in all_tid2items:
                    all_tid2items[tid] = []
                all_tid2items[tid].append((fid, score, tid2fid[tid] == fid))

        # Save fid2items to database
        for fid, items in fid_stream.items():
            fid_stream[fid] = items

        # Save tid2items to database
        for tid, items in all_tid2items.items():
            tid_stream[tid] = items

    print(f"Saved fid2items to {fid2items_db_fpath}")
    print(f"Saved tid2items to {tid2items_db_fpath}")
