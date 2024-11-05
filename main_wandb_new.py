import click
import yaml
import os
import random
import time

import numpy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

# Import utility modules for data and model handling
from utils import criterion_utils, data_utils, model_utils

# Set random seeds for reproducibility
torch.manual_seed(0)
random.seed(0)
numpy.random.seed(0)


# Main training function
def train_model(conf, sweep, run_id=None):
    """
    Train the model with specified configurations.

    :param conf: The configuration dictionary.
    :param run_id: Optional run ID for resuming a previous run.
    """
    # Initialize Weights & Biases run
    if sweep:
        # NOTE Will default initialize the repo name as the project name
        wandb.init()
        print(wandb.config)
        if wandb.config:
            for key, value in wandb.config.items():
                ls = key.split("-")
                conf[f"{ls[0]}"][f"{ls[1]}"][f"{ls[2]}"] = value

    else:
        project_name = conf.get("wandb_conf", {})["project"]
        if run_id:
            wandb.init(id=run_id, resume="must", project=project_name)
        else:
            wandb.init(project=project_name, config=conf)

    # print(conf)
    # Load data and parameter configurations, applying sweep overrides if available
    data_conf = conf["data_conf"]
    param_conf = conf["param_conf"]

    # Override with wandb.config for hyperparameters during sweeps
    param_conf["batch_size"] = param_conf["batch_size"]
    param_conf["num_epoch"] = param_conf["num_epoch"]

    # Initialize and load training and validation data
    train_ds = data_utils.load_data(data_conf["train_data"])
    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=param_conf["batch_size"],
        shuffle=True,
        collate_fn=data_utils.collate_fn,
    )

    val_ds = data_utils.load_data(data_conf["val_data"])
    val_dl = DataLoader(
        dataset=val_ds,
        batch_size=param_conf["batch_size"],
        shuffle=True,
        collate_fn=data_utils.collate_fn,
    )

    # Initialize model, objective, optimizer, and learning rate scheduler
    # Load initial model params
    model_params = conf[param_conf["model"]]

    # Initialize the model with the updated params
    model = model_utils.init_model(model_params, train_ds.text_vocab)
    # print(model)  # Print the initialized model architecture

    # Objective and optimizer initialization
    obj_params = conf["criteria"][param_conf["criterion"]]
    objective = getattr(criterion_utils, obj_params["name"], None)(
        **obj_params["args"]
    )

    optim_params = conf[param_conf["optimizer"]]
    optimizer_args = optim_params["args"].copy()  # Copy the args dictionary

    optimizer = getattr(optim, optim_params["name"], None)(
        model.parameters(), **optimizer_args
    )

    lr_params = conf[param_conf["lr_scheduler"]]
    lr_scheduler = getattr(
        optim.lr_scheduler, lr_params["name"], "ReduceLROnPlateau"
    )(optimizer, **lr_params["args"])

    # Early stopping object
    early_stopping_params = conf[param_conf["early_stopper"]]
    early_stopping = criterion_utils.EarlyStopping(
        **early_stopping_params["args"]
    )

    wandb.watch(model)

    # Load model from checkpoint if specified and resuming
    if run_id:
        checkpoint = torch.load(os.path.join(wandb.run.dir, "checkpoint"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 1

    # Training loop
    max_epoch = param_conf["num_epoch"]
    save_interval = 100  # Save checkpoint every 20 epochs

    best_val_loss = float("inf")

    # Perform training
    for epoch in range(start_epoch, max_epoch + 1):
        if epoch > 0:
            # Perform model training
            model_utils.tqdm_train(
                model, train_dl, objective, optimizer, epoch, max_epoch
            )

        # Perform validation
        train_loss = model_utils.eval(model, train_dl, objective)
        val_loss = model_utils.eval(model, val_dl, objective)

        # Log metrics to Weights & Biases
        epoch_results = {
            "train_obj": train_loss,
            "val_obj": val_loss,
            "stop_metric": val_loss,
        }
        # Reduce learning rate w.r.t validation loss
        lr_scheduler.step(epoch_results["stop_metric"])

        wandb.log(
            epoch_results,
            step=epoch,
        )

        # Check early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            save_path = os.path.join("./z_ckpts", f"{wandb.run.id}")
            os.makedirs(save_path, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(save_path, f"checkpoint_epoch_{epoch}"),
            )
            break

        # Save the model checkpoint locally every save_interval epochs and at the end
        if epoch % save_interval == 0 or epoch == max_epoch:
            # part = wandb.run.dir.split("/wandb/")[1].split("/files")[0]
            # # Here run name is id, as we are extracting it from the run dirs.
            # chars, date, run_name = part.split("-")
            save_path = os.path.join("./z_ckpts", f"{wandb.run.id}")
            os.makedirs(save_path, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(save_path, f"checkpoint_epoch_{epoch}"),
            )

        # Perform data reloading (pair bootstrapping)
        train_ds = data_utils.load_data(data_conf["train_data"])
        train_dl = DataLoader(
            dataset=train_ds,
            batch_size=param_conf["batch_size"],
            shuffle=True,
            collate_fn=data_utils.collate_fn,
        )

        val_ds = data_utils.load_data(data_conf["val_data"])
        val_dl = DataLoader(
            dataset=val_ds,
            batch_size=param_conf["batch_size"],
            shuffle=True,
            collate_fn=data_utils.collate_fn,
        )

    # Save the final model weights to wandb
    # torch.save(model.state_dict(), os.path.join(wandb.run.dir, "final_model_weights.pth"))
    # wandb.save("final_model_weights.pth")

    # Finish the Weights & Biases run
    wandb.finish()


def load_config(config_path):
    """Load the sweep configuration from a YAML file."""
    with open(config_path, "r") as stream:
        return yaml.safe_load(stream)


@click.command()
@click.option(
    "--sweep",
    required=False,
    default=False,
    is_flag=True,
    help="Run the WandB sweep.",
)
def main(sweep):
    print(sweep)
    # Load the main configuration from the YAML file
    conf_num = 0
    # normal_conf_path = f"./conf_yamls/cap_{conf_num}_conf.yaml"
    normal_conf_path = f"conf_yamls/base_configs/cap_0_new_conf.yaml"
    sweep_conf_path = (
        f"conf_yamls/sweep_1_oct_29_2024_60_runs/hyperparam_sweep.yaml"
    )
    conf = load_config(normal_conf_path)
    sweep_config = load_config(sweep_conf_path)
    # Check if resuming a previous run
    resume_run_id = conf.get("resume_run_id")

    if sweep:
        # If the sweep flag is set, run the WandB sweep
        sweep_id = wandb.sweep(
            sweep=sweep_config, project="dcase2023-audio-retrieval"
        )
        wandb.agent(
            sweep_id,
            function=lambda: train_model(conf, sweep, resume_run_id),
            count=60,
        )  # Adjust count as needed
    else:
        # Train the model
        train_model(conf, resume_run_id)


if __name__ == "__main__":
    wandb.login()  # Log in to WandB before starting the sweep
    main()
