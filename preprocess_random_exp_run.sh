#!/bin/bash
export $(xargs<.env)

echo "Preprocessing for Replication Factor 1"

# echo "Preprocessing Audio and Doing Random Selection"
# python random_selection_based_preprocessing/01_agen_clotho_dataset.py

# echo "Creating Audio Logmels"
# python random_selection_based_preprocessing/02_agen_multiprocessing_audio_logmel.py

echo "Creating sbert embeddings"
python random_selection_based_preprocessing/03_agen_sbert_embeddings.py

# echo "Model Weight Transfer"
# python random_selection_based_preprocessing/04_cnn14_transfer.py