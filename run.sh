#!/bin/bash
export $(xargs<.env)

# Run the script in sequence
echo "Running 1x"
python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/base_configs/cap_1x_conf.yaml > 1x.log 2>&1

echo "Running 2x"
python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/base_configs/cap_2x_conf.yaml > 2x.log 2>&1

echo "Running 3x"
python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/base_configs/cap_3x_conf.yaml > 3x.log 2>&1

echo "Running 4x"
python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/base_configs/cap_4x_conf.yaml > 4x.log 2>&1

echo "Running 5x"
python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/base_configs/cap_5x_conf.yaml > 5x.log 2>&1
