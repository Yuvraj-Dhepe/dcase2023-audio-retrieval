import dbm
import json
import os
import shelve
from dbm import dumb
from tqdm import tqdm
import torch
import yaml
import wandb
import nltk
from utils import criterion_utils, data_utils, model_utils

# Set the default module for dbm to 'dumb'
dbm._defaultmod = dumb
dbm._modules = {"dbm.dumb": dumb}
# Load English stopwords from nltk
stopwords = nltk.corpus.stopwords.words("english")

cap_col_num, run_num = '3x', 3
# Load configuration from conf.yaml
with open(f"./conf_yamls/cap_{cap_col_num}_conf.yaml", "rb") as stream:
    conf = yaml.full_load(stream)

# Extract wandb configuration
wandb_conf = conf.get("wandb_conf", {})

# Get the latest wandb run directory
api = wandb.Api()
runs = api.runs(wandb_conf['project'])  # Replace with your actual project name
latest_run = runs[len(runs) - run_num]

print(f"\nInitializing Run ID: {latest_run.id} && Run Name: {latest_run.name}\n")

# Construct the checkpoint directory path
ckp_fpath = os.path.join("./z_ckpts", latest_run.id)

# Load data
data_conf = conf["data_conf"]
train_ds = data_utils.load_data(data_conf["train_data"], train=False)
val_ds = data_utils.load_data(data_conf["val_data"], train=False)
eval_ds = data_utils.load_data(data_conf["eval_data"], train=False)

# Restore model checkpoint
param_conf = conf["param_conf"]
model_params = conf[param_conf["model"]]
obj_params = conf["criteria"][param_conf["criterion"]]
model = model_utils.init_model(model_params, train_ds.text_vocab)
model = model_utils.restore(model, os.path.join(ckp_fpath,f"checkpoint_epoch_{param_conf['num_epoch']}"))

# Control GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"USING: {device}")
model.to(device)

# Set the model to evaluation mode
model.eval()

# Batch size for processing data
batch_size = 1024  # Adjust based on your GPU memory

# Iterate through datasets (train, val, eval)
for name, ds in zip(["train", "val", "eval"], [train_ds, val_ds, eval_ds]):
    # Initialize a dictionary to store text embeddings
    text2vec = {}

    # Encode text data in batches
    for idx in tqdm(range(0, len(ds.text_data), batch_size), desc=f"Encoding text for {name} dataset"):
        # Get a batch of text data
        batch_data = ds.text_data.iloc[idx:min(idx + batch_size, len(ds.text_data))]

        # Initialize a list to store embedded text vectors for the batch
        batch_text_vecs = []

        # Iterate through the batch of text data
        for item in batch_data.itertuples():
            # Embed text data based on text level (word or sentence)
            if ds.text_level == "word":
                text_vec = torch.as_tensor([ds.text_vocab(key) for key in item.tokens if key not in stopwords])
                text_vec = model.text_branch(torch.unsqueeze(text_vec, dim=0).to(device))[0]  # Embed text data
                batch_text_vecs.append(torch.unsqueeze(text_vec, dim=0).to(device))
            elif ds.text_level == "sentence":
                text_vec = torch.as_tensor([ds.text_vocab(item.tid)]).to(device)
                text_vec = model.text_branch(torch.unsqueeze(text_vec, dim=0))[0]  # Embed text data
                batch_text_vecs.append(torch.unsqueeze(text_vec, dim=0))

        # Update text2vec dictionary with embedded text vectors for the batch
        text2vec.update({item.tid: vec for item, vec in zip(batch_data.itertuples(), batch_text_vecs)})

    # Compute pairwise cross-modal scores
    score_fpath = os.path.join(ckp_fpath, f"{name}_xmodal_scores.db")
    with shelve.open(filename=score_fpath, flag="n", protocol=2) as stream:
        # Iterate through unique audio file identifiers
        for fid in tqdm(ds.text_data["fid"].unique(), desc=f"Computing cross-modal scores for {name} dataset"):
            # Initialize a dictionary to store scores for the current audio file
            group_scores = {}

            # Encode audio data
            audio_vec = torch.as_tensor(ds.audio_data[fid][()]).to(device)
            audio_vec = torch.unsqueeze(audio_vec, dim=0)
            audio_embed = model.audio_branch(audio_vec)[0] # 300 is it's shape

            # For a single fid, we will calculate the scores for all tids
            # Calculate scores in batches
            for i in range(0, len(text2vec), batch_size):
                # Get batch of text IDs
                batch_text_ids = list(text2vec.keys())[i:min(i + batch_size, len(text2vec))]

                # Create a tensor of embedded text vectors for the batch
                batch_text_embeds = torch.stack([text2vec[tid] for tid in batch_text_ids]).to(device)

                # Reshape for matrix multiplication (batch_size, embedding_dim)
                batch_text_embeds = batch_text_embeds.reshape(-1, batch_text_embeds.shape[-1])

                # Calculate cross-modal scores for the batch
                xmodal_scores = criterion_utils.score(audio_embed, batch_text_embeds, obj_params["args"].get("dist", "dot_product"))

                # Update group_scores with calculated scores
                for j, tid in enumerate(batch_text_ids):
                    group_scores[tid] = xmodal_scores[j].item()

            # Save scores for the current audio file
            stream[fid] = group_scores
            #{'fid':{'tid1':'score between tid1 and fid', 'tid2':'score between tid2 and fid'}, 'fid2':{'tid1':'score between tid1 and fid', 'tid2':'score between tid2 and fid2'}}

    print("Save", score_fpath)