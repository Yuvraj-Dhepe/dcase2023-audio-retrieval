#!/bin/bash
export $(xargs<.env)
# Execute the first Python file
echo "Training in Progress"
python wandb_new.py

echo "Calculating Audio-Text Scores"
# Execute the second Python file
python postprocessing/scores_new.py

echo "Evaluation in Progress"
# Execute the third Python file
python postprocessing/retrieval_new.py