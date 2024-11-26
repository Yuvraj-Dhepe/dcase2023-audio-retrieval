import os
import numpy as np
import torch
import wandb
from postprocessing import modular_scores_wandb
from ast import literal_eval
from tqdm import tqdm
import dbm
from dbm import dumb
import shelve
import pandas as pd

from postprocessing.modular_retrieval_split_wandb import (
    process_db,
    remove_files_with_extensions,
)


def postprocess_scores(conf, model_weights_folder):
    data_conf = conf["data_conf"]
    param_conf = conf["param_conf"]

    # Load the datasets
    train_ds, val_ds, eval_ds, eval_obj_ds = (
        modular_scores_wandb.load_datasets(
            data_conf, eval_obj_batch_size=param_conf["batch_size"]
        )
    )

    # Load the model
    model = modular_scores_wandb.initialize_model(
        conf, train_ds, model_weights_folder
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"USING: {device}")
    model.to(device)
    model.eval()

    # Encode the text data
    batch_size = 1024  # Adjust based on GPU memory
    for name, ds in zip(["val", "eval"], [val_ds, eval_ds]):
        text2vec = modular_scores_wandb.encode_text_data(
            ds, model, device, batch_size
        )
        modular_scores_wandb.compute_and_save_scores(
            name,
            ds,
            model,
            text2vec,
            device,
            batch_size,
            model_weights_folder,
            conf,
        )


def postprocess_data_split(conf, model_weights_folder):

    ckp_fpath = model_weights_folder
    data_conf = conf["data_conf"]

    # Extract data configuration
    data_conf = conf["data_conf"]

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

        # Load cross-modal scores from the shelve file
        score_fpath = os.path.join(ckp_fpath, f"{name}_xmodal_scores.db")

        # Initialize shelve databases for storage
        fid2items_db_fpath = os.path.join(
            ckp_fpath, f"{name}_fid_2_items_latest.db"
        )
        tid2items_db_fpath = os.path.join(
            ckp_fpath, f"{name}_tid_2_items_latest.db"
        )

        with shelve.open(score_fpath) as stream, shelve.open(
            fid2items_db_fpath, "c"
        ) as fid_stream, shelve.open(tid2items_db_fpath, "c") as tid_stream:

            # Create an empty dictionary to store results for tid2items
            all_tid2items = {}

            # Iterate through the audio file IDs in the shelve file
            for fid, group_scores in tqdm(
                stream.items(), desc="Processing Items"
            ):

                # Audio2Text retrieval: Create a list of tuples (tid, score, is_relevant) for each audio file
                fid_stream[fid] = [
                    (tid, group_scores[tid], tid2fid[tid] == fid)
                    for tid in group_scores
                ]

                # Text2Audio retrieval: Create a list of tuples (fid, score, is_relevant) for each text ID
                for tid, score in group_scores.items():
                    if tid not in all_tid2items:
                        all_tid2items[tid] = []
                    all_tid2items[tid].append(
                        (fid, score, tid2fid[tid] == fid)
                    )

            # Save fid2items to database
            for fid, items in fid_stream.items():
                fid_stream[fid] = items

            # Save tid2items to database
            for tid, items in all_tid2items.items():
                tid_stream[tid] = items

        print(f"Saved fid2items to {fid2items_db_fpath}")
        print(f"Saved tid2items to {tid2items_db_fpath}")


def postprocess_score_retrieval(conf, model_weights_folder):
    ckp_fpath = model_weights_folder
    data_conf = conf["data_conf"]

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

    # NOTE: Emptying the folders removing post processing files except csv files
    # Specify the folder and extensions
    extensions = [".dat", ".bak", ".dir"]
    remove_files_with_extensions(ckp_fpath, extensions)
