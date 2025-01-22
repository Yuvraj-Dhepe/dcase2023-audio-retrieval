#!/bin/bash
# Load environment variables from .env
export $(xargs < .env)

# Ensure logs directory exists
mkdir -p logs

# Generate date-time for log filename
date_time=$(date '+%Y-%m-%d_%H-%M-%S')

# echo "Transfer the model weights"
# python random_selection_based_preprocessing/04_cnn14_transfer.py

echo "Running split 1"
python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/sweep_configs/best_config.yaml > logs/"${date_time}_Original.log" 2>&1

# echo "Running 1x"
# python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/ez_confs/cap_1x_conf.yaml > logs/EZ1x.log 2>&1

# echo "Running 2x"
# python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/ez_confs/cap_2x_conf.yaml > logs/EZ2x.log 2>&1

# echo "Running 3x"
# python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/ez_confs/cap_3x_conf.yaml > logs/EZ3x.log 2>&1

# echo "Running 4x"
# python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/ez_confs/cap_4x_conf.yaml > logs/EZ4x.log 2>&1

# echo "Running 5x"
# python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/ez_confs/cap_5x_conf.yaml > logs/EZ5x.log 2>&1