#!/bin/bash
export $(xargs<.env)

# Run the script in sequence
echo "Running 1x"
python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/ez_confs/cap_1x_conf.yaml > logs/EZ1x.log 2>&1

echo "Running 2x"
python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/ez_confs/cap_2x_conf.yaml > logs/EZ2x.log 2>&1

echo "Running 3x"
python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/ez_confs/cap_3x_conf.yaml > logs/EZ3x.log 2>&1

echo "Running 4x"
python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/ez_confs/cap_4x_conf.yaml > logs/EZ4x.log 2>&1

echo "Running 5x"
python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/ez_confs/cap_5x_conf.yaml > logs/EZ5x.log 2>&1