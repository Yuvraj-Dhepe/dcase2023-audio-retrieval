import os
import shelve
import pandas as pd
from tqdm import tqdm
from ast import literal_eval
import dbm
from dbm import dumb
import wandb
import yaml
import click

dbm._defaultmod = dumb
dbm._modules = {"dbm.dumb": dumb}


def load_config(config_path):
    """
    Load configuration from a YAML file.
    """
    with open(config_path, "rb") as stream:
        conf = yaml.full_load(stream)
    return conf


def get_latest_run(project_name, run_num=None, run_id=None):
    """
    Get the latest run from Weights & Biases, either by run number or run ID.
    """
    api = wandb.Api()
    runs = api.runs(project_name)

    if run_id:
        latest_run = next(run for run in runs if run.id == run_id)
    elif run_num:
        latest_run = runs[len(runs) - run_num]
    else:
        raise ValueError("Either 'run_num' or 'run_id' must be specified.")

    return latest_run


def process_data(config_path, run_id=None, run_num=None):
    """
    Process the datasets for the specified run number or run ID.
    """
    conf = load_config(config_path=config_path)
    # Extract WandB configuration
    wandb_conf = conf.get("wandb_conf", {})

    # Get the latest WandB run directory
    latest_run = get_latest_run(wandb_conf["project"], run_num, run_id)

    # Construct the checkpoint directory path
    print(
        f"\nInitializing Run ID: {latest_run.id} && Run Name: {latest_run.name}\n"
    )

    ckp_fpath = os.path.join("./z_ckpts", latest_run.id)

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


@click.command()
@click.option(
    "--config",
    type=str,
    required=True,
    help="Path to the YAML configuration file.",
)
@click.option(
    "--run_num", type=int, default=None, help="Run number for experiment."
)
@click.option(
    "--run_id", type=str, default=None, help="Run ID for experiment."
)
def main(config, run_num, run_id):
    """
    CLI entry point for processing data.
    """

    process_data(config_path=config, run_num=run_num, run_id=run_id)


if __name__ == "__main__":
    main()
