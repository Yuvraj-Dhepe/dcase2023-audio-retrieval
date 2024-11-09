import os
import shelve
import numpy as np
import pandas as pd
import glob
from ast import literal_eval
import yaml
import wandb
from tqdm import tqdm
import click


def load_config(config_path):
    """Loads the configuration from a YAML file."""
    with open(config_path, "rb") as stream:
        conf = yaml.full_load(stream)
    return conf


def initialize_wandb_run(conf, run_num=1, run_id=None):
    """Initializes WandB run based on provided configuration."""
    wandb_conf = conf.get("wandb_conf", {})
    api = wandb.Api()

    if run_id:
        run = api.run(f"{wandb_conf['project']}/{run_id}")
    else:
        runs = api.runs(wandb_conf["project"])
        run = runs[len(runs) - run_num]

    print(f"\nInitializing Run ID: {run.id} && Run Name: {run.name}\n")
    wandb.init(project=wandb_conf["project"], id=run.id, resume="must")
    # wandb.init(project=wandb_conf["project"], id=run.id)
    return run.id


def measure_single_item(qid, items, dataset_name, way):
    """Calculates retrieval metrics (recall@K, mAP@10) for a single query ID."""
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
    """Processes a shelve database and calculates retrieval metrics."""
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


def score_retrieval(config_path, run_num=None, run_id=None):
    """Main function to process datasets and calculate metrics."""
    conf = load_config(config_path)
    run_id = initialize_wandb_run(conf, run_num, run_id)

    # Construct the checkpoint directory path
    ckp_fpath = os.path.join("./z_ckpts", run_id)
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
    wandb.finish()


def remove_files_with_extensions(folder_path, extensions):
    """
    Removes all files in the specified folder that match the given extensions.

    Args:
        folder_path (str): The path to the folder.
        extensions (list): List of file extensions to be removed.
    """
    for ext in extensions:
        # Create a search pattern for each extension
        search_pattern = os.path.join(folder_path, f"*{ext}")
        # Use glob to find files matching the pattern
        for file_path in glob.glob(search_pattern):
            try:
                os.remove(file_path)
                # print(f"Removed: {file_path}")
            except OSError as e:
                print(f"Error removing {file_path}: {e}")


@click.command()
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True),
    help="Path to the configuration YAML file.",
)
@click.option(
    "--run_num",
    default=None,
    type=int,
    help="Run number to fetch. Ignored if --run-id is provided.",
)
@click.option(
    "--run_id", default=None, type=str, help="Specific Run ID to use."
)
def run_script(config, run_num, run_id):
    """CLI entry point."""
    score_retrieval(config_path=config, run_num=run_num, run_id=run_id)


if __name__ == "__main__":
    run_script()
