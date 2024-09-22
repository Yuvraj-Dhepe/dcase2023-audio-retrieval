
import os
import shelve
import pandas as pd
from tqdm import tqdm
from ast import literal_eval
import dbm
from dbm import dumb
import wandb
import yaml
import gc
from collections import defaultdict

dbm._defaultmod = dumb
dbm._modules = {"dbm.dumb": dumb}

# Load configuration from conf.yaml
cap_col_num, run_num = 1, 5

with open(f"./conf_yamls/exp_{cap_col_num}x_conf.yaml", "rb") as stream:
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
for name in data_conf.keys():
    params = data_conf[name]
    name = name.replace("_data", "")

    # Load text data in chunks to minimize memory usage
    text_fpath = os.path.join(params["dataset"], params["text_data"])
    text_data_iterator = pd.read_csv(text_fpath, converters={"tokens": literal_eval}, chunksize=1000)
    print("\n\nLoaded", text_fpath)

    # Initialize mappings with defaultdict to minimize memory usage
    tid2fid, tid2text, fid2fname = {}, {}, {}

    # Process data chunk by chunk
    for chunk in tqdm(text_data_iterator, desc="Processing Text Data"):
        for idx, item in chunk.iterrows():
            tid2fid[item["tid"]] = item["fid"]
            tid2text[item["tid"]] = item["text"]
            fid2fname[item["fid"]] = item["fname"]

        # Free chunk from memory
        del chunk
        gc.collect()

    # Load cross-modal scores from the shelve file in small batches
    score_fpath = os.path.join(ckp_fpath, f"{name}_xmodal_scores.db")

    # Initialize shelve databases for storage
    fid2items_db_fpath = os.path.join(ckp_fpath, f"{name}_fid_2_items_optimized.db")
    tid2items_db_fpath = os.path.join(ckp_fpath, f"{name}_tid_2_items_optimized.db")

    with shelve.open(score_fpath) as stream, shelve.open(fid2items_db_fpath, 'c') as fid_stream, shelve.open(tid2items_db_fpath, 'c') as tid_stream:

        # Iterate through the audio file IDs in the shelve file
        for fid, group_scores in tqdm(stream.items(), desc="Processing Items"):

            # Audio2Text retrieval: Write directly to shelve to avoid keeping large objects in memory
            fid_items = [(tid, group_scores[tid], tid2fid[tid] == fid) for tid in group_scores]
            fid_stream[fid] = fid_items  # Save directly into shelve

            # Text2Audio retrieval: Process in small batches to avoid large memory usage
            # NOTE: This is extremely slow as .db files are hell in frequent writes.
            for tid, score in group_scores.items():
                if tid not in tid_stream:
                    tid_stream[tid] = []
                # Append new data to the list
                temp_list = tid_stream[tid]  # Fetch the existing list
                temp_list.append((fid, score, tid2fid[tid] == fid))  # Append to the list
                tid_stream[tid] = temp_list  # Save the updated list back to shelve


        # Free up memory after processing
        gc.collect()
    print(f"Saved fid2items to {fid2items_db_fpath}")
    print(f"Saved tid2items to {tid2items_db_fpath}")
