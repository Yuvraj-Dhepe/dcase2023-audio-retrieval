#!/bin/bash
export $(xargs<.env)
# Execute the first Python file
echo "Post-processing 1"
python postprocessing/01_scores_wandb_new.py

echo "Post-processing 2"
python postprocessing/02_xmodel_data_split.py

echo "Post-processing 3"
python postprocessing/03_retrieval_split_wandb_new.py