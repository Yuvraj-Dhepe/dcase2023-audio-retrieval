import base64
import glob
import os
import pickle
import re
from ast import literal_eval
import hashlib
import pandas as pd
from mutagen.wave import WAVE
from tqdm import tqdm

### NOTE: From each dataset dirs, the audios in the `audio_split` folders and {split}_captions.csv files will be used to create {split}_text.csv

# - Takes in dataset directory, and generates hash-based fid mapping for the fnames for each split. Produces a single audio_info.pkl for all splits
# - Takes in the {split}_captions.csv file containing 5 captions for each .wav file, and maps a hash-based tid for the text. Produces {split}_text.csv files for each data split
# - Takes in the {split}_text.csv files and generates data statistics for all splits. Produces a single file of vocab_info.pkl


global_params = {
    # "dataset_dirs": ["./data/Clotho"], # Can use commented combinations as well
    "dataset_dirs": ["./data/Clotho", "./data/Clotho_caption_5"],
    # "dataset_dirs": ["./data/Clotho", "./data/Clotho_caption_2", "./data/Clotho_caption_3"]
    "audio_splits": ["development", "validation", "evaluation"],
    "text_files": ["development_captions.csv", "validation_captions.csv", "evaluation_captions.csv"]
}

def process_audio_data(dataset_dirs, audio_splits):
    """
    Processes audio data to extract file IDs (fids) and durations.

    Args:
        dataset_dirs (list): List of directories containing audio data.
        audio_splits (list): List of audio splits (e.g., "development", "validation").

    Returns:
        dict: A dictionary mapping audio splits to dictionaries of fid-to-filename and durations.
    """
    audio_fid2fname, audio_durations = {}, {}

    for split in audio_splits:
        fid2fname, durations = {}, []
        # Use all dataset_dirs for development, else only the first one, because we are using both the synthetic and original audio data only for development split
        if split == "development":
            used_dataset_dirs = dataset_dirs  # Use all directories
        else:
            used_dataset_dirs = [dataset_dirs[0]] # Use only the first directory

        for dataset_dir in used_dataset_dirs:
            audio_dir = os.path.join(dataset_dir, split)

            for fpath in tqdm(glob.glob(r"{}/*.wav".format(audio_dir)), desc=f"Processing audio files in {split}"):
                try:
                    clip = WAVE(fpath)

                    if clip.info.length > 0.0:
                        fid = hashlib.md5(fpath.encode()).hexdigest()
                        fname = os.path.basename(fpath)

                        fid2fname[fid] = fname
                        durations.append(clip.info.length)
                except:
                    print("Error audio file:", fpath)

        assert len(fid2fname) == len(durations)

        audio_fid2fname[split] = fid2fname
        audio_durations[split] = durations

    return {"audio_fid2fname": audio_fid2fname, "audio_durations": audio_durations}

def process_text_data(dataset_dirs, audio_splits, text_files, audio_fid2fname):
    """
    Processes text data to generate text IDs (tids) and tokens.

    Args:
        dataset_dirs (list): List of directories containing text data.
        audio_splits (list): List of audio splits.
        text_files (list): List of text files corresponding to audio splits.
        audio_fid2fname (dict): Dictionary mapping audio splits to fid-to-filename mappings.
    """
    for split, text_fname in zip(audio_splits, text_files):

        fid2fname = audio_fid2fname[split]
        stripped_fname2fid = {fid2fname[fid].strip(" "): fid for fid in fid2fname}

        text_fpath = os.path.join(dataset_dirs[-1], text_fname)
        text_data = pd.read_csv(text_fpath)

        text_rows = []

        for i in tqdm(text_data.index, desc=f"Processing text data in {split}"):
            fname = text_data.iloc[i].fname
            raw_text = text_data.iloc[i].raw_text
            text = text_data.iloc[i].text

            fid = stripped_fname2fid[fname]
            tid = hashlib.md5((fid + text).encode()).hexdigest()

            tokens = [t for t in re.split(r"\s", text) if len(t) > 0]

            text_rows.append([tid, fid, fid2fname[fid], raw_text, text, tokens])

        text_rows = pd.DataFrame(data=text_rows, columns=["tid", "fid", "fname", "raw_text", "text", "tokens"])

        text_fpath = os.path.join(dataset_dirs[-1], f"{split}_text.csv")
        text_rows.to_csv(text_fpath, index=False)
        print("Save", text_fpath)
        print('='*90)

def generate_data_statistics(dataset_dirs, audio_splits, audio_fid2fname):
    """
    Generates data statistics including vocabulary, word bags, and split information.

    Args:
        dataset_dirs (list): List of directories containing text data.
        audio_splits (list): List of audio splits.
        audio_fid2fname (dict): Dictionary mapping audio splits to fid-to-filename mappings.
    """
    vocabulary = set()
    word_bags = {}
    split_infos = {}

    for split in audio_splits:

        fid2fname = audio_fid2fname[split]

        text_fpath = os.path.join(dataset_dirs[-1], f"{split}_text.csv")
        text_data = pd.read_csv(text_fpath, converters={"tokens": literal_eval})

        num_clips = len(fid2fname)
        num_captions = text_data.tid.size

        bag = []
        for tokens in tqdm(text_data["tokens"], desc=f"Generating data statistics for {split}"):
            bag.extend(tokens)
            vocabulary = vocabulary.union(tokens)

        num_words = len(bag)
        word_bags[split] = bag
        split_infos[split] = {
            "num_clips": num_clips,
            "num_captions": num_captions,
            "num_words": num_words
        }

    return {"vocabulary": vocabulary, "word_bags": word_bags, "split_infos": split_infos}

# %% Main execution

if __name__ == "__main__":
    # Process audio data
    audio_data = process_audio_data(global_params["dataset_dirs"], global_params["audio_splits"])

    # Save audio info
    audio_info = os.path.join(global_params["dataset_dirs"][-1], "audio_info.pkl")
    with open(audio_info, "wb") as store:
        pickle.dump(audio_data, store)
    print("Save audio info to", audio_info)
    print('='*90)

    # Process text data
    process_text_data(global_params["dataset_dirs"], global_params["audio_splits"],global_params["text_files"], audio_data["audio_fid2fname"])

    # Generate data statistics
    data_statistics = generate_data_statistics(global_params["dataset_dirs"], global_params["audio_splits"],audio_data["audio_fid2fname"])

    # Save vocabulary
    vocab_info = os.path.join(global_params["dataset_dirs"][-1], "vocab_info.pkl")
    with open(vocab_info, "wb") as store:
        pickle.dump(data_statistics, store)
    print("Save vocabulary info to", vocab_info)
    print('='*90)