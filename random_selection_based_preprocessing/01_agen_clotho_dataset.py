import os
import glob
import pickle
import re
import hashlib
import random
from ast import literal_eval
import pandas as pd
from mutagen.wave import WAVE
from tqdm import tqdm


def process_audio_data(
    dataset_dirs, audio_splits, replication_factor, seed=None
):
    """
    Processes audio data to extract file IDs (fids) and durations, with random selection of synthetic copies
    for the development split.

    :param dataset_dirs: List of dataset directories (for original and synthetic files).
    :param audio_splits: List of audio splits (e.g., "development", "validation", "evaluation").
    :param replication_factor: Number of synthetic copies to select for each original audio file.
    :param seed: Seed for random number generator to ensure reproducibility.
    :return: Dictionary containing audio file IDs, file names, and durations.
    """

    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Initialize dictionaries to store filenames and durations for each split
    audio_fid2fname, audio_durations = {}, {}

    # Iterate over each split (development, validation, evaluation)
    for split in audio_splits:
        fid2fname, durations = {}, []

        # For the development split, use all dataset directories (for random selection)
        if split == "development":
            used_dataset_dirs = dataset_dirs
        else:
            # For other splits, use only the original dataset (first directory)
            used_dataset_dirs = [dataset_dirs[0]]

        original_files = []  # List to store original audio files
        synthetic_files = (
            {}
        )  # Dictionary to store synthetic audio files based on original file names

        # Process each directory in the used dataset directories
        for dataset_dir in used_dataset_dirs:
            audio_dir = os.path.join(dataset_dir, split)

            # Use glob to get all WAV files in the directory and process them with tqdm for progress
            for fpath in tqdm(
                glob.glob(r"{}/*.wav".format(audio_dir)),
                desc=f"Processing audio files in {split}",
            ):
                try:
                    clip = WAVE(fpath)  # Extract audio info using mutagen
                    if (
                        clip.info.length > 0.0
                    ):  # Ensure the audio file has valid length
                        fid = hashlib.md5(
                            fpath.encode()
                        ).hexdigest()  # Generate unique ID for each file
                        fname = os.path.basename(fpath)
                        if (
                            "_cap_" not in fname
                        ):  # If file is original (not synthetic)
                            original_files.append(
                                (fid, fname, clip.info.length)
                            )  # Add original file info
                        else:
                            # Group synthetic files by base name (without '_cap_x.wav' suffix)
                            base_name = fname[:-10]  # Removes `_cap_x.wav`
                            if base_name not in synthetic_files:
                                synthetic_files[base_name] = []
                            synthetic_files[base_name].append(
                                (fid, fname, clip.info.length)
                            )  # Add synthetic file info
                except Exception as e:
                    print(f"Error processing audio file: {fpath} | Error: {e}")

        # Handle the development split: randomly select synthetic files based on replication_factor
        if split == "development":
            selected_fid2fname = (
                {}
            )  # Dictionary to store selected file IDs and file names
            selected_durations = []  # List to store selected file durations

            # Iterate over each original file
            for original_fid, original_fname, original_duration in tqdm(
                original_files, desc="Selecting synthetic copies"
            ):
                # Add the original file to the selected files
                selected_fid2fname[original_fid] = original_fname
                selected_durations.append(original_duration)

                # Get synthetic candidates based on the original file's base name
                synthetic_candidates = synthetic_files.get(
                    original_fname[:-4], []
                )
                random.shuffle(
                    synthetic_candidates
                )  # Shuffle the synthetic candidates randomly

                # Select 'replication_factor' number of synthetic files
                selected_candidates = synthetic_candidates[:replication_factor]

                # Add the selected synthetic files to the result
                for (
                    synthetic_fid,
                    synthetic_fname,
                    synthetic_duration,
                ) in selected_candidates:
                    selected_fid2fname[synthetic_fid] = synthetic_fname
                    selected_durations.append(synthetic_duration)

            # Store the selected files and durations for the development split
            audio_fid2fname[split] = selected_fid2fname
            audio_durations[split] = selected_durations
        else:
            # For validation and evaluation splits, directly use all original files
            for fid, fname, duration in original_files:
                fid2fname[fid] = fname
                durations.append(duration)

            # Store the filenames and durations for the split
            audio_fid2fname[split] = fid2fname
            audio_durations[split] = durations

    return {
        "audio_fid2fname": audio_fid2fname,
        "audio_durations": audio_durations,
    }


def process_text_data(text_files, audio_fid2fname, output_dir):
    """
    Processes text data to generate text IDs (tids) and tokens.

    :param text_files: Dictionary of text file paths for each split.
    :param audio_fid2fname: Dictionary mapping file IDs to filenames for each split.
    :param output_dir: Directory to save processed text data.
    """
    # Iterate over each split and corresponding text file
    for split, text_fpath in text_files.items():
        fid2fname = audio_fid2fname["audio_fid2fname"][
            split
        ]  # Get the file ID to filename mapping for the split
        stripped_fname2fid = {
            fid2fname[fid].strip(" "): fid for fid in fid2fname
        }  # Map stripped filenames to file IDs

        text_data = pd.read_csv(text_fpath)  # Load text data from CSV

        text_rows = []  # List to store processed text rows

        # Process each row in the text data with tqdm to track progress
        for i in tqdm(
            text_data.index, desc=f"Processing text data in {split}"
        ):
            fname = text_data.iloc[i].fname
            raw_text = text_data.iloc[i].raw_text
            text = text_data.iloc[i].text

            # Selecting only the fid's for the randomly selected audios
            fid = stripped_fname2fid.get(
                fname
            )  # Get the corresponding file ID
            if fid:
                tid = hashlib.md5(
                    (fid + text).encode()
                ).hexdigest()  # Generate a unique text ID (tid)
                tokens = [
                    t for t in re.split(r"\s", text) if len(t) > 0
                ]  # Tokenize the text
                text_rows.append(
                    [tid, fid, fid2fname[fid], raw_text, text, tokens]
                )  # Append processed text data

        # Convert the text rows to a DataFrame and save as CSV
        text_rows_df = pd.DataFrame(
            data=text_rows,
            columns=["tid", "fid", "fname", "raw_text", "text", "tokens"],
        )
        text_fpath = os.path.join(output_dir, f"{split}_text.csv")
        text_rows_df.to_csv(text_fpath, index=False)
        print(f"Saved {text_fpath}")
        print("=" * 90)


def generate_data_statistics(audio_splits, audio_fid2fname, output_dir):
    """
    Generates data statistics including vocabulary, word bags, and split information.

    :param audio_splits: List of audio splits (e.g., "development", "validation", "evaluation").
    :param audio_fid2fname: Dictionary containing audio file ID-to-filename mappings for each split.
    :param output_dir: Directory to save the data statistics.
    :return: Dictionary containing vocabulary, word bags, and split information.
    """
    vocabulary = set()  # Set to store unique vocabulary
    word_bags = {}  # Dictionary to store word bags for each split
    split_infos = {}  # Dictionary to store info about each split

    # Iterate over each split
    for split in audio_splits:
        fid2fname = audio_fid2fname["audio_fid2fname"][
            split
        ]  # Get file ID to filename mapping for the split

        # Load processed text data
        text_fpath = os.path.join(output_dir, f"{split}_text.csv")
        text_data = pd.read_csv(
            text_fpath, converters={"tokens": literal_eval}
        )

        num_clips = len(fid2fname)  # Number of clips (audio files)
        num_captions = text_data.tid.size  # Number of captions (text entries)

        # Generate word bag (list of all words in the text data)
        bag = []
        for tokens in tqdm(
            text_data["tokens"], desc=f"Generating data statistics for {split}"
        ):
            bag.extend(tokens)
            vocabulary = vocabulary.union(
                tokens
            )  # Update vocabulary with unique tokens

        num_words = len(bag)  # Total number of words in the split
        word_bags[split] = bag  # Store word bag for the split
        split_infos[split] = {
            "num_clips": num_clips,
            "num_captions": num_captions,
            "num_words": num_words,
        }

    return {
        "vocabulary": vocabulary,
        "word_bags": word_bags,
        "split_infos": split_infos,
    }


if __name__ == "__main__":
    global_params = {
        "dataset_dirs": [
            "./data/Clotho",
            "./data/ez_clotho_caption_1",
            "./data/ez_clotho_caption_2",
            "./data/ez_clotho_caption_3",
            "./data/ez_clotho_caption_4",
            "./data/ez_clotho_caption_5",
        ],
        "audio_splits": ["development", "validation", "evaluation"],
        "text_files": {
            "development": "data/extended_development_captions.csv",
            "validation": "data/Clotho/validation_captions.csv",
            "evaluation": "data/Clotho/evaluation_captions.csv",
        },
    }

    replication_factor = 5  # User-defined value for how many times to pick from synthetic copies (1 to 5)

    output_dir = f"./data/EZexp_{replication_factor}"
    os.makedirs(output_dir, exist_ok=True)

    # Process audio data
    audio_data = process_audio_data(
        global_params["dataset_dirs"],
        global_params["audio_splits"],
        replication_factor,
        seed=24,
    )

    # Save audio info
    audio_info = os.path.join(output_dir, "audio_info.pkl")
    with open(audio_info, "wb") as store:
        pickle.dump(audio_data, store)
    print(f"Saved audio info to {audio_info}")
    print("=" * 90)

    # Process text data
    process_text_data(global_params["text_files"], audio_data, output_dir)

    # Generate data statistics
    data_statistics = generate_data_statistics(
        global_params["audio_splits"], audio_data, output_dir
    )

    # Save vocabulary
    vocab_info = os.path.join(output_dir, "vocab_info.pkl")
    with open(vocab_info, "wb") as store:
        pickle.dump(data_statistics, store)
    print(f"Saved vocabulary info to {vocab_info}")
    print("=" * 90)
