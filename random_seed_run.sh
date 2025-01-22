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
python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/random_seed_confs_88/cap_1x_conf.yaml > logs/"${date_time}_random_seed_88_split_1.log" 2>&1

echo "Running split 2"
python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/random_seed_confs_88/cap_2x_conf.yaml > logs/"${date_time}_random_seed_88_split_2.log" 2>&1

echo "Running split 3"
python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/random_seed_confs_88/cap_3x_conf.yaml > logs/"${date_time}_random_seed_88_split_3.log" 2>&1

echo "Running split 4"
python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/random_seed_confs_88/cap_4x_conf.yaml > logs/"${date_time}_random_seed_88_split_4.log" 2>&1

echo "Running split 5"
python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/random_seed_confs_88/cap_5x_conf.yaml > logs/"${date_time}_random_seed_88_split_5.log" 2>&1

echo "Running split 6"
python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/random_seed_confs_88/cap_6x_conf.yaml > logs/"${date_time}_random_seed_88_split_6.log" 2>&1

echo "Running split 7"
python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/random_seed_confs_88/cap_7x_conf.yaml > logs/"${date_time}_random_seed_88_split_7.log" 2>&1

echo "Running split 8"
python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/random_seed_confs_88/cap_8x_conf.yaml > logs/"${date_time}_random_seed_88_split_8.log" 2>&1

echo "Running split 9"
python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/random_seed_confs_88/cap_9x_conf.yaml > logs/"${date_time}_random_seed_88_split_9.log" 2>&1

echo "Running split 10"
python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/random_seed_confs_88/cap_10x_conf.yaml > logs/"${date_time}_random_seed_88_split_10.log" 2>&1

