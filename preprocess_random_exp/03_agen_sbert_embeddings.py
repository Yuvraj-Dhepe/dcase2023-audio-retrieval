import os
import pickle
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

global_params = {
    "dataset_dir": "./data/Clotho_caption_5",
    "audio_splits": ["development", "validation", "evaluation"]
}

model_name = "sbert"
# Create model and move it to the device
# By default model will be put on GPU if available.
model = SentenceTransformer('all-mpnet-base-v2')
def generate_text_embeddings(output_dir, audio_splits, model):
    """
    Generates text embeddings for captions using a SentenceTransformer model.

    Args:
        dataset_dir (str): Path to the dataset directory.
        audio_splits (list): List of audio splits (e.g., "development", "validation").
        model (SentenceTransformer): Pre-trained SentenceTransformer model.

    Returns:
        dict: A dictionary mapping text IDs (tids) to their corresponding embeddings.
    """
    text_embeds = {}

    for split in audio_splits:
        text_fpath = os.path.join(output_dir, f"{split}_text.csv")
        text_data = pd.read_csv(text_fpath)

        for i in tqdm(text_data.index, desc=f"Generating embeddings for {split}"):
            tid = text_data.loc[i].tid
            raw_text = text_data.loc[i].raw_text

            text_embeds[tid] = model.encode(raw_text)

    return text_embeds

def save_text_embeddings(embed_fpath, text_embeds):
    """
    Saves text embeddings to a pickle file.

    Args:
        embed_fpath (str): Path to the output pickle file.
        text_embeds (dict): Dictionary mapping text IDs (tids) to their embeddings.
    """
    with open(embed_fpath, "wb") as stream:
        pickle.dump(text_embeds, stream)

    print("Save text embeddings to", embed_fpath)

# Main execution
if __name__ == "__main__":
    output_dir = './data/exp_5'
    text_embeddings = generate_text_embeddings(output_dir, global_params["audio_splits"], model)

    # Save text embeddings
    embed_fpath = os.path.join(output_dir, f"{model_name}_embeds.pkl")
    save_text_embeddings(embed_fpath, text_embeddings)