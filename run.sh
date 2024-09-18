#!/bin/bash
export $(xargs<.env)
# Execute the first Python file
echo "Preprocessing 1"
python preprocess_random_exp/01_agen_clotho_dataset.py

echo "Preprocessing 2"
python preprocess_random_exp/02_agen_multiprocessing_audio_logmel.py

echo "Preprocessing 3"
python preprocess_random_exp/03_agen_sbert_embeddings.py

echo "Preprocessing 4"
python preprocess_random_exp/04_cnn14_transfer.py

echo "Training"
python main_wandb_new.py