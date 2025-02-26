#!/bin/bash
export $(xargs<.env)

echo "Preprocessing for Replication Factor 3"

echo "Preprocessing Audio and Doing Random Selection"
python random_seed_based_experiment/01_agen_clotho_dataset.py

echo "Creating Audio Logmels"
python random_seed_based_experiment/02_agen_multiprocessing_audio_logmel.py

echo "Creating sbert embeddings"
python random_seed_based_experiment/03_agen_sbert_embeddings.py