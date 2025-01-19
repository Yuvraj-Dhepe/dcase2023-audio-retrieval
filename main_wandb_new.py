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
from postprocessing import sweep_postprocessing
from utils import criterion_utils, data_utils, model_utils

# Set random seeds for reproducibility
torch.manual_seed(24)
random.seed(24)
numpy.random.seed(24)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Leads to no optimization by pytorch but determinism


# Main training function
def train_model(conf, sweep, run_id):
    """
    Train the model with specified configurations.

    :param conf: The configuration dictionary.
    :param run_id: Optional run ID for resuming a previous run.
    """
    # Initialize Weights & Biases run
    if sweep:
        # NOTE Will default initialize the repo name as the project name
        wandb.init(reinit=True)
        if wandb.config:
            for key, value in wandb.config.items():
                ls = key.split("-")
                # NOTE: Criteria has a depth of 4, others only 3
                if ls[0] == "criteria":
                    conf[f"{ls[0]}"][f"{ls[1]}"][f"{ls[2]}"][
                        f"{ls[3]}"
                    ] = value
                    continue
                conf[f"{ls[0]}"][f"{ls[1]}"][f"{ls[2]}"] = value
    else:
        project_name = conf.get("wandb_conf", {}).get("project")
        try:
            if run_id:
                # Attempt to resume with the provided run_id
                wandb.init(
                    id=run_id, resume="must", project=project_name, config=conf
                )
                # Check if this run already has metrics for all epochs
                run = wandb.Api().run(f"{project_name}/{run_id}")
                max_epoch = conf["param_conf"]["num_epoch"]
                logged_epochs = [row["epoch"] for row in run.history()]
                if max_epoch in logged_epochs:
                    print(
                        f"Run {run_id} is already completed for all epochs. Skipping..."
                    )
                    return  # Exit early if the run is fully logged
            else:
                # Start a fresh run if no run_id is provided
                wandb.init(project=project_name, config=conf)
        except Exception as e:
            print(f"Error resuming run: {e}. Starting a new run.")
            wandb.init(project=project_name, config=conf)

    # print(conf)
    # Load data and parameter configurations, applying sweep overrides if available
    data_conf = conf["data_conf"]
    param_conf = conf["param_conf"]

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
        # TODO: Have a better logic here
        # checkpoint = torch.load(os.path.join(wandb.run.dir, "checkpoint"))
        # model.load_state_dict(checkpoint["model_state_dict"])
        # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # start_epoch = checkpoint["epoch"]
        start_epoch = 1
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
        save_path = os.path.join("./z_ckpts", f"{wandb.run.id}")
        if epoch % save_interval == 0 or epoch == max_epoch:
            # part = wandb.run.dir.split("/wandb/")[1].split("/files")[0]
            # # Here run name is id, as we are extracting it from the run dirs.
            # chars, date, run_name = part.split("-")
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

    # NOTE: Calculate the eval_mAP once after the epochs are finished, only for hpt optimization in sweep
    if 1:
        sweep_postprocessing.postprocess_scores(
            conf=conf, model_weights_folder=save_path
        )

        sweep_postprocessing.postprocess_data_split(
            conf=conf, model_weights_folder=save_path
        )

        sweep_postprocessing.postprocess_score_retrieval(
            conf=conf, model_weights_folder=save_path
        )

    wandb.finish()


@click.group
def cli():
    """CLI for managing model training and WandB sweeps."""


@cli.command
@click.option(
    "--base_conf_path",
    required=True,
    help="Path to the normal configuration YAML file.",
)
@click.option(
    "--sweep_conf_path",
    required=True,
    help="Path to the sweep configuration YAML file.",
)
@click.option(
    "--run_count",
    required=False,
    default=100,
    help="Number of sweeps to perform",
)
@click.option(
    "--project_name",
    default="dcase2023-audio-retrieval",
    required=False,
    help="The project under which the sweep should be created",
)
def sweep(base_conf_path, sweep_conf_path, run_count, project_name):
    """Run a new WandB sweep."""
    base_config = load_config(base_conf_path)
    sweep_config = load_config(sweep_conf_path)

    sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)
    wandb.agent(
        sweep_id,
        function=lambda: train_model(
            conf=base_config, sweep=True, run_id=False
        ),
        count=run_count,
    )


@cli.command
@click.option(
    "--agent_id",
    required=True,
    help="Agent ID for resuming a previously run agent in the sweep.",
)
@click.option(
    "--base_conf_path",
    required=True,
    help="Path to the normal configuration YAML file.",
)
@click.option(
    "--sweep_conf_path",
    required=True,
    help="Path to the sweep configuration YAML file.",
)
@click.option(
    "--run_count",
    required=False,
    default=100,
    help="Number of sweeps to perform",
)
@click.option(
    "--project_name",
    default="dcase2023-audio-retrieval",
    required=False,
    help="The project under which the sweep should be created",
)
def agent(agent_id, base_conf_path, sweep_conf_path, run_count, project_name):
    """Resume a previously run WandB sweep agent."""
    base_config = load_config(base_conf_path)
    sweep_config = load_config(sweep_conf_path)

    wandb.agent(
        agent_id,
        function=lambda: train_model(
            conf=base_config, sweep=True, run_id=False
        ),
        count=run_count,
        project=project_name,
    )


@cli.command
@click.option(
    "--run_id",
    default=None,
    required=False,
    help="Run ID for resuming a previously logged run.",
)
@click.option(
    "--base_conf_path",
    required=True,
    help="Path to the normal configuration YAML file.",
)
def resume_or_new_run(run_id, base_conf_path):
    """Resume a previously logged WandB run."""
    base_config = load_config(base_conf_path)
    train_model(base_config, sweep=False, run_id=run_id)


def load_config(config_path):
    """Load a configuration YAML file."""
    import yaml

    with open(config_path, "r") as stream:
        return yaml.safe_load(stream)


if __name__ == "__main__":
    wandb.login()
    cli()
