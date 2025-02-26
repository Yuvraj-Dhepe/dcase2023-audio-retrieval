import yaml
import os
import random
import time

import numpy
import ray
import torch
import torch.optim as optim
from ray import tune
from ray.tune.progress_reporter import CLIReporter
from ray.tune.stopper import TrialPlateauStopper
from torch.utils.data import DataLoader

# Import utility modules for data and model handling
from utils import criterion_utils, data_utils, model_utils

# Set random seeds for reproducibility
torch.manual_seed(0)
random.seed(0)
numpy.random.seed(0)


# Main trial execution function
def exec_trial(conf, ckp_dir=None):
    """
    Execute a trial for hyperparameter tuning with ray tune.

    :param conf: The configuration dictionary for this trial
    :param ckp_dir: Optional checkpoint directory to load model from
    """
    # Load data and parameter configurations
    data_conf = conf["data_conf"]
    param_conf = conf["param_conf"]

    # Create training and validation data loaders
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
    model_params = conf[param_conf["model"]]
    model = model_utils.init_model(model_params, train_ds.text_vocab)
    print(model)  # Print the initialized model architecture

    obj_params = conf["criteria"][param_conf["criterion"]]
    objective = getattr(criterion_utils, obj_params["name"], None)(
        **obj_params["args"]
    )

    optim_params = conf[param_conf["optimizer"]]
    optimizer = getattr(optim, optim_params["name"], None)(
        model.parameters(), **optim_params["args"]
    )

    lr_params = conf[param_conf["lr_scheduler"]]
    lr_scheduler = getattr(
        optim.lr_scheduler, lr_params["name"], "ReduceLROnPlateau"
    )(optimizer, **lr_params["args"])

    # Load model from checkpoint if specified
    if ckp_dir is not None:
        model_state, optimizer_state = torch.load(
            os.path.join(ckp_dir, "checkpoint")
        )
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # Define maximum number of epochs for this trial
    max_epoch = param_conf["num_epoch"] + 1

    # Perform trial execution
    breakpoint()
    for epoch in numpy.arange(0, max_epoch):
        if epoch > 0:
            # Perform model training
            model_utils.tqdm_train(model, train_dl, objective, optimizer)

        # Initialize epoch results
        epoch_results = {}

        # Perform model evaluation
        epoch_results["train_obj"] = model_utils.eval(
            model, train_dl, objective
        )
        epoch_results["val_obj"] = model_utils.eval(model, val_dl, objective)

        # Determine stop metric (e.g., validation loss)
        epoch_results["stop_metric"] = epoch_results["val_obj"]

        # Reduce learning rate w.r.t validation loss
        lr_scheduler.step(epoch_results["stop_metric"])

        # Save the model to the trial directory
        with tune.checkpoint_dir(step=epoch) as ckp_dir:
            path = os.path.join(ckp_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        # Report the current epoch results back to ray tune
        tune.report(**epoch_results)

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


# Main entry point for hyperparameter tuning
if __name__ == "__main__":
    # Load the main configuration from the YAML file
    with open("conf.yaml", "rb") as stream:
        conf = yaml.full_load(stream)

    # Configure the ray tune cluster
    # NOTE: temp_dir must be absolute PATH
    # Configure the ray tune cluster
    ray_conf = conf["ray_conf"]
    ray_conf["init_args"][
        "_temp_dir"
    ] = "/tmp/ray_temp"  # Use an absolute path
    ray.init(**ray_conf["init_args"])
    trial_stopper = getattr(
        tune.stopper, ray_conf["trial_stopper"], TrialPlateauStopper
    )(**ray_conf["stopper_args"])
    trial_reporter = CLIReporter(
        max_report_frequency=60, print_intermediate_tables=True
    )

    # Define metrics for the CLI reporter
    for metric in ["train_obj", "val_obj", "stop_metric"]:
        trial_reporter.add_metric_column(metric=metric)

    breakpoint()

    def trial_name_creator(trial):
        """
        Generate the name of a trial.

        :param trial: The ray tune trial
        """
        trial_name = "_".join([conf["param_conf"]["model"], trial.trial_id])
        return trial_name

    def trial_dirname_creator(trial):
        """
        Generate the directory name for a trial.

        :param trial: The ray tune trial
        """
        trial_dirname = "_".join([time.strftime("%Y-%m-%d"), trial.trial_id])
        return trial_dirname

    # Execute trials
    analysis = tune.run(
        run_or_experiment=exec_trial,
        name=conf["trial_series"],
        stop=trial_stopper,
        config=conf,
        resources_per_trial={"cpu": 1, "gpu": 1},
        num_samples=1,
        storage_path=conf["trial_base"],
        keep_checkpoints_num=3,
        checkpoint_score_attr="min-stop_metric",
        verbose=1,
        progress_reporter=trial_reporter,
        log_to_file=True,
        trial_name_creator=trial_name_creator,
        trial_dirname_creator=trial_dirname_creator,
        fail_fast=False,
        reuse_actors=True,
        raise_on_failed_trial=True,
    )

    # Find the best trial based on the stop metric
    best_trial = analysis.get_best_trial(
        metric=ray_conf["stopper_args"]["metric"],
        mode=ray_conf["stopper_args"]["mode"],
        scope="all",
    )
    print("Best trial:", best_trial.trial_id)

    # Find the best checkpoint
    best_ckp = analysis.get_best_checkpoint(
        trial=best_trial,
        metric=ray_conf["stopper_args"]["metric"],
        mode=ray_conf["stopper_args"]["mode"],
    )
    print("Best checkpoint:", best_ckp)
