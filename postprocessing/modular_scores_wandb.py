import dbm
import glob
import os
import shelve
import click
import pandas as pd
from dbm import dumb
from tqdm import tqdm
import torch
import yaml
import wandb
import nltk
from utils import criterion_utils, data_utils, model_utils
from torch.utils.data import DataLoader

# Set the default module for dbm to 'dumb'
dbm._defaultmod = dumb
dbm._modules = {"dbm.dumb": dumb}
# Load English stopwords from nltk
stopwords = nltk.corpus.stopwords.words("english")


def run_experiment(config_path, run_num=None, run_id=None, params_csv=None):
    """
    Initialize the model, load data, and compute cross-modal scores.
    """
    conf = load_config(config_path)

    # If a parameter CSV is provided, update the configuration from the CSV
    if params_csv:
        conf = update_config_from_csv(conf, params_csv, run_id)

    # print(conf)

    wandb_conf = conf.get("wandb_conf", {})

    # Initialize experiment with run ID or run number
    latest_run = get_latest_run(wandb_conf["project"], run_num, run_id)
    wandb.init(
        project=wandb_conf["project"], id=run_id, resume="must", config=conf
    )
    ckp_fpath = os.path.join("./z_ckpts", latest_run.id)

    print(
        f"\nInitializing Run ID: {latest_run.id} && Run Name: {latest_run.name}\n"
    )

    # Load data and model
    data_conf = conf["data_conf"]
    train_ds, val_ds, eval_ds, eval_obj_ds = load_datasets(
        data_conf, eval_obj_batch_size=conf["param_conf"]["batch_size"]
    )
    model = initialize_model(conf, train_ds, ckp_fpath)

    # Set device and move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"USING: {device}")
    model.to(device)

    # Set model to evaluation mode
    model.eval()
    batch_size = 1024  # Adjust based on GPU memory

    # NOTE To calculate eval_loss objective
    param_conf = conf["param_conf"]
    obj_params = conf["criteria"][param_conf["criterion"]]

    objective = getattr(criterion_utils, obj_params["name"], None)(
        **obj_params["args"]
    )
    # Iterate through datasets and encode text data
    for name, ds in zip(["val", "eval"], [val_ds, eval_ds]):
        text2vec = encode_text_data(ds, model, device, batch_size)
        compute_and_save_scores(
            name, ds, model, text2vec, device, batch_size, ckp_fpath, conf
        )
    # NOTE: Log the value eval_obj value to the run
    eval_obj = get_eval_obj(model, eval_obj_ds, objective)
    wandb.log({"after_train_eval_obj": eval_obj})
    wandb.finish()


def get_eval_obj(model, data_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion.to(device=device)
    eval_loss, eval_steps = 0.0, 0
    with torch.inference_mode():
        # Wrap the data_loader with tqdm
        with tqdm(
            data_loader,
            unit="batch",
            desc="Evaluating loss objective for eval set",
        ) as tepoch:
            for batch_idx, data in enumerate(tepoch, 0):
                item_batch, audio_batch, text_batch = data

                audio_batch = audio_batch.to(device)
                text_batch = text_batch.to(device)

                audio_embeds, text_embeds = model(audio_batch, text_batch)
                loss = criterion(audio_embeds, text_embeds, item_batch)
                eval_loss += loss.cpu().numpy()
                eval_steps += 1

                # Update tqdm progress bar (optional)
                tepoch.set_postfix(loss=loss.item())

    return eval_loss / max(eval_steps, 1)


def load_config(config_path):
    """
    Load configuration from a YAML file.
    """
    with open(config_path, "rb") as stream:
        conf = yaml.full_load(stream)
    return conf


def update_config_from_csv(conf, csv_path, run_id):
    """
    Update the configuration dictionary based on matching parameters from a CSV file.
    Only updates parameters present in the original configuration.
    """
    df = pd.read_csv(csv_path)
    if "id" not in df.columns:
        raise ValueError(
            "CSV file must contain an 'id' column to match the run ID."
        )

    # Filter the row that matches the provided run ID
    row = df[df["id"] == run_id]
    if row.empty:
        raise ValueError(
            f"No matching entry for id {run_id} found in CSV file."
        )

    # Iterate over the columns to update the config

    for col in row.columns:
        if col == "id":  # Skip the id column
            continue
        col_parts = col.split("-")

        try:
            # Convert to int if "fc_units" is in the column name
            value = (
                int(row[col].values[0])
                if "fc_units" in col_parts
                else row[col].values[0]
            )
            if "temperature" in col_parts:
                conf[f"{col_parts[0]}"][f"{col_parts[1]}"][f"{col_parts[2]}"][
                    f"{col_parts[3]}"
                ] = value
                continue
            conf[col_parts[0]][col_parts[1]][col_parts[2]] = value

        except KeyError:
            print(
                f"Warning: '{col}' not found in the configuration structure."
            )
    print("Configuration updated successfully.")
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


def load_datasets(data_conf, eval_obj_batch_size=32):
    """
    Load datasets based on the configuration.
    """
    train_ds = data_utils.load_data(data_conf["train_data"], train=False)
    val_ds = data_utils.load_data(data_conf["val_data"], train=False)
    eval_ds = data_utils.load_data(data_conf["eval_data"], train=False)
    eval_obj_ds = data_utils.load_data(data_conf["val_data"])
    eval_obj_ds = DataLoader(
        dataset=val_ds,
        batch_size=eval_obj_batch_size,
        shuffle=True,
        collate_fn=data_utils.collate_fn,
    )
    return train_ds, val_ds, eval_ds, eval_obj_ds


def initialize_model(conf, train_ds, ckp_fpath):
    """
    Initialize the model and restore the first available checkpoint file starting with 'checkpoint_epoch'.
    """
    param_conf = conf["param_conf"]
    model_params = conf[param_conf["model"]]
    model = model_utils.init_model(model_params, train_ds.text_vocab)

    # Search for the first file with the prefix 'checkpoint_epoch' in the checkpoint path
    checkpoint_files = sorted(
        glob.glob(os.path.join(ckp_fpath, "checkpoint_epoch*"))
    )

    if checkpoint_files:
        checkpoint_path = checkpoint_files[0]  # Use the first file in the list
        print(f"Restoring model from checkpoint: {checkpoint_path}")
        model = model_utils.restore(model, checkpoint_path)
    else:
        raise FileNotFoundError(
            f"No checkpoint files found with prefix 'checkpoint_epoch' in {ckp_fpath}"
        )

    return model


def encode_text_data(ds, model, device, batch_size):
    """
    Encode text data for the specified dataset.
    """
    text2vec = {}
    with torch.inference_mode():
        for idx in tqdm(
            range(0, len(ds.text_data), batch_size),
            desc=f"Encoding text for {ds} dataset",
        ):
            batch_data = ds.text_data.iloc[
                idx : min(idx + batch_size, len(ds.text_data))
            ]
            batch_text_vecs = []
            for item in batch_data.itertuples():
                if ds.text_level == "word":
                    text_vec = torch.as_tensor(
                        [
                            ds.text_vocab(key)
                            for key in item.tokens
                            if key not in stopwords
                        ]
                    )
                    text_vec = model.text_branch(
                        torch.unsqueeze(text_vec, dim=0).to(device)
                    )[0]
                    batch_text_vecs.append(
                        torch.unsqueeze(text_vec, dim=0).to(device)
                    )
                elif ds.text_level == "sentence":
                    text_vec = torch.as_tensor([ds.text_vocab(item.tid)]).to(
                        device
                    )
                    text_vec = model.text_branch(
                        torch.unsqueeze(text_vec, dim=0)
                    )[0]
                    batch_text_vecs.append(torch.unsqueeze(text_vec, dim=0))

            text2vec.update(
                {
                    item.tid: vec
                    for item, vec in zip(
                        batch_data.itertuples(), batch_text_vecs
                    )
                }
            )
    return text2vec


def compute_and_save_scores(
    name, ds, model, text2vec, device, batch_size, ckp_fpath, conf
):
    """
    Compute pairwise cross-modal scores and save them using shelve.
    """
    param_conf = conf["param_conf"]
    obj_params = conf["criteria"][param_conf["criterion"]]
    score_fpath = os.path.join(ckp_fpath, f"{name}_xmodal_scores.db")

    with shelve.open(filename=score_fpath, flag="n", protocol=2) as stream:
        with torch.inference_mode():
            for fid in tqdm(
                ds.text_data["fid"].unique(),
                desc=f"Computing cross-modal scores for {name} dataset",
            ):
                group_scores = {}
                audio_vec = torch.as_tensor(ds.audio_data[fid][()]).to(device)
                audio_vec = torch.unsqueeze(audio_vec, dim=0)
                audio_embed = model.audio_branch(audio_vec)[0]

                for i in range(0, len(text2vec), batch_size):
                    batch_text_ids = list(text2vec.keys())[
                        i : min(i + batch_size, len(text2vec))
                    ]
                    batch_text_embeds = torch.stack(
                        [text2vec[tid] for tid in batch_text_ids]
                    ).to(device)
                    batch_text_embeds = batch_text_embeds.reshape(
                        -1, batch_text_embeds.shape[-1]
                    )

                    xmodal_scores = criterion_utils.score(
                        audio_embed,
                        batch_text_embeds,
                        obj_params["args"].get("dist", "dot_product"),
                    )

                    for j, tid in enumerate(batch_text_ids):
                        group_scores[tid] = xmodal_scores[j].item()

                stream[fid] = group_scores
    print("Save", score_fpath)


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
@click.option(
    "--params_csv", type=str, default=None, help="Path to parameters CSV file."
)
def main(config, run_num, run_id, params_csv):
    """
    CLI entry point for the experiment.
    """
    run_experiment(
        config_path=config,
        run_num=run_num,
        run_id=run_id,
        params_csv=params_csv,
    )


if __name__ == "__main__":
    main()
