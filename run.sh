#!/bin/bash
# Load environment variables from .env
export $(xargs < .env)

# Ensure logs directory exists
mkdir -p logs

# Generate date-time for log filename
date_time=$(date '+%Y-%m-%d_%H-%M-%S')

echo "Running Baseline with AudioGen Synthetic 1x Subset"
python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/base_configs/cap_1x_conf.yaml > logs/audiogen_1x.log 2>&1

echo "Running Baseline with AudioGen Synthetic 2x Subset"
python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/base_configs/cap_2x_conf.yaml > logs/audiogen_2x.log 2>&1

echo "Running Baseline with AudioGen Synthetic 3x Subset"
python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/base_configs/cap_3x_conf.yaml > logs/audiogen_3x.log 2>&1

echo "Running Baseline with AudioGen Synthetic 4x Subset"
python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/base_configs/cap_4x_conf.yaml > logs/audiogen_4x.log 2>&1

echo "Running Baseline with AudioGen Synthetic 5x Subset"
python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/base_configs/cap_5x_conf.yaml > logs/audiogen_5x.log 2>&1

echo "Running Baseline with EzAudio Synthetic 1x Subset"
python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/ez_confs/cap_1x_conf.yaml > logs/EzAudio_1x.log 2>&1

echo "Running Baseline with EzAudio Synthetic 2x Subset"
python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/ez_confs/cap_2x_conf.yaml > logs/EzAudio_2x.log 2>&1

echo "Running Baseline with EzAudio Synthetic 3x Subset"
python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/ez_confs/cap_3x_conf.yaml > logs/EzAudio_3x.log 2>&1

echo "Running Baseline with EzAudio Synthetic 4x Subset"
python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/ez_confs/cap_4x_conf.yaml > logs/EzAudio_4x.log 2>&1

echo "Running Baseline with EzAudio Synthetic 5x Subset"
python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/ez_confs/cap_5x_conf.yaml > logs/audiogen_5x.log 2>&1