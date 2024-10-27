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
cap_col_num, run_num = "1x", 5

with open(f"./conf_yamls/cap_{cap_col_num}_conf.yaml", "rb") as stream:
    conf = yaml.full_load(stream)

# Extract WandB configuration
wandb_conf = conf.get("wandb_conf", {})

# Get the latest WandB run directory
api = wandb.Api()
runs = api.runs(wandb_conf["project"])  # Replace with your actual project name
latest_run = runs[len(runs) - run_num]

# Construct the checkpoint directory path
print(
    f"\nInitializing Run ID: {latest_run.id} && Run Name: {latest_run.name}\n"
)

ckp_fpath = os.path.join("./z_ckpts", latest_run.id)

# Extract data configuration
data_conf = conf["data_conf"]


# Function to batch save items to shelve
def batch_save_to_shelve(shelve_db, data_batch, append_mode=False):
    """Saves batches to shelve, with the option to append (without overwriting)."""
    for key, value in data_batch.items():
        if append_mode:
            # If key already exists, append to the existing list
            if key in shelve_db:
                existing_list = shelve_db[key]
                existing_list.extend(value)
                shelve_db[key] = existing_list
            else:
                shelve_db[key] = value
        else:
            shelve_db[key] = value
    data_batch.clear()  # Clear the batch after saving to shelve


# Iterate through datasets
for name in data_conf.keys():
    params = data_conf[name]
    name = name.replace("_data", "")

    # Load text data in chunks to minimize memory usage
    text_fpath = os.path.join(params["dataset"], params["text_data"])
    text_data_iterator = pd.read_csv(
        text_fpath, converters={"tokens": literal_eval}, chunksize=5000
    )
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
    fid2items_db_fpath = os.path.join(
        ckp_fpath, f"{name}_fid_2_items_optimized.db"
    )
    tid2items_db_fpath = os.path.join(
        ckp_fpath, f"{name}_tid_2_items_optimized.db"
    )

    # Delete the files if they exist
    if os.path.exists(fid2items_db_fpath):
        os.remove(fid2items_db_fpath)
    if os.path.exists(tid2items_db_fpath):
        os.remove(tid2items_db_fpath)

    with shelve.open(score_fpath) as stream, shelve.open(
        fid2items_db_fpath, "c"
    ) as fid_stream, shelve.open(tid2items_db_fpath, "c") as tid_stream:

        ### Phase 1: Process and save fid_stream ###
        fid_batch = {}
        fid_batch_size = 3000  # Adjust fid batch size

        # Process fid_stream in batches
        for fid, group_scores in tqdm(
            stream.items(), desc="Processing fid_stream"
        ):
            fid_batch[fid] = [
                (tid, group_scores[tid], tid2fid[tid] == fid)
                for tid in group_scores
            ]

            # Once the fid batch size is reached, flush to shelve
            if len(fid_batch) >= fid_batch_size:
                batch_save_to_shelve(fid_stream, fid_batch)
                gc.collect()

        # Flush remaining fid_batch items
        batch_save_to_shelve(fid_stream, fid_batch)
        print(f"Saved fid2items to {fid2items_db_fpath}")

        ### Phase 2: Process and save tid_stream ###
        tid_batch = {}
        tid_batch_size = (
            1000  # Adjust tid batch size (smaller for less memory usage)
        )

        # NOTE: This is again damn slow
        # Re-iterate over stream for tid_stream processing
        for fid, group_scores in tqdm(
            stream.items(), desc="Processing tid_stream"
        ):
            for tid, score in group_scores.items():
                if tid not in tid_batch:
                    tid_batch[tid] = []
                tid_batch[tid].append((fid, score, tid2fid[tid] == fid))

            # Once the tid batch size is reached, flush to shelve
            if len(tid_batch) >= tid_batch_size:
                batch_save_to_shelve(tid_stream, tid_batch, append_mode=True)
                gc.collect()

        # Flush remaining tid_batch items
        batch_save_to_shelve(tid_stream, tid_batch, append_mode=True)
        print(f"Saved tid2items to {tid2items_db_fpath}")

    gc.collect()
