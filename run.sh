#!/bin/bash
export $(xargs<.env)
# Execute the first Python file
echo "Preprocessing 1"
python preprocessing/01_agen_clotho_dataset.py

echo "Preprocessing 2"
python preprocessing/02_agen_multiprocessing_audio_logmel.py

echo "Preprocessing 3"
python preprocessing/03_agen_sbert_embeddings.py

echo "Preprocessing 4"
python preprocessing/04_cnn14_transfer.py

echo "Training"
python main_wandb_new.py