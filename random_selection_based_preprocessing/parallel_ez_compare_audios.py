import click
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
from scipy.spatial.distance import euclidean
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from models import audio_encoders
from utils import audio_encoder_layer_map as ael
import yaml


def load_model(
    model_conf="./conf_yamls/base_configs/cap_0_conf.yaml",
    weights_path="./pretrained_models_weights/cnn14.pth",
):
    model_params = yaml.safe_load(open(model_conf))["DualEncoderModel"]
    # Load a base audio PANNS model for comparisons
    # Initiate CNN14 model
    ael.transfer_cnn_14_params(
        weights_path=weights_path,
        output_path=f"{os.path.dirname(weights_path)}/CNN14_300.pth",
        layer_name_mapping=ael.cnn14_transfer_layer_mapping(),
        out_dim=model_params["audio_enc"]["out_dim"],
        conv_dropout=model_params["audio_enc"]["conv_dropout"],
        fc_dropout=model_params["audio_enc"]["fc_dropout"],
        fc_units=model_params["audio_enc"]["fc_units"],
    )
    cnn14_encoder = audio_encoders.CNN14Encoder(**model_params["audio_enc"])

    # Load a base audio PANNS model for comparisons
    # Initiate CNN14 model

    # Load pretrained parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(
        f"{os.path.dirname(weights_path)}/CNN14_300.pth", weights_only=True
    )
    cnn14_encoder.load_state_dict(state_dict)
    cnn14_encoder.to(device)
    cnn14_encoder.eval()

    return cnn14_encoder


cnn14_encoder = load_model()


# Warning utility Function
def handle_warnings(func, name="", *args, **kwargs):
    """Handle warnings globally for any function."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # Always trigger warnings
        result = func(*args, **kwargs)  # Call the function
        if w:
            for warning in w:
                print(
                    f"Warning for {name}: {warning.message}"
                )  # Print warning message
    return result


# Input Creator & Normalizer Function
def log_mel_spectrogram(
    y,
    sample_rate=44100,
    window_length_secs=0.040,
    hop_length_secs=0.020,
    num_mels=64,
    log_offset=np.spacing(1),
    name="",
    method="",
):
    """
    Convert waveform to a log magnitude mel-frequency spectrogram.
    """
    window_length = int(round(sample_rate * window_length_secs))
    hop_length = int(round(sample_rate * hop_length_secs))
    fft_length = 2 ** int(np.ceil(np.log(window_length) / np.log(2.0)))

    mel_spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=sample_rate,
        n_fft=fft_length,
        hop_length=hop_length,
        win_length=window_length,
        n_mels=num_mels,
    )
    # Ensure no negative or zero values before applying log
    mel_spectrogram = mel_spectrogram + log_offset

    try:
        log_mel = handle_warnings(
            np.log, name, mel_spectrogram
        )  # Log mel spectrogram with warning handling
    except ValueError as e:
        print(f"Error calculating log-mel spectrogram: {e}")
        return None

    # NOTE: Return transpose for model method as that's what is used in training
    if method == "model":
        return log_mel.T  # [time, mel]
    else:
        return log_mel


def convert_format(log_mel):
    """
    Convert log-mel spectrogram from [time, mel] to [mel, time] and vice versa.

    :param log_mel: Log-mel spectrogram.
    :return: Converted log-mel spectrogram.
    """
    return log_mel.T


def min_max_normalize_spectrogram(spectrogram, audio_name=""):
    """Normalize the spectrogram values between 0 and 1.
    If all scores are high, you might want to normalize the spectrograms before comparison. Normalization can help reduce the influence of volume differences.
    You can apply techniques like min-max scaling or z-score normalization to your spectrograms before calculating the similarity score.
    """
    min_val = np.min(spectrogram)
    max_val = np.max(spectrogram)

    if max_val == min_val:
        print(f"Warning: Spectrogram for '{audio_name}' has zero variance.")
        return spectrogram  # Return as-is to avoid divide by zero

    try:
        return handle_warnings(
            lambda s: (s - min_val) / (max_val - min_val),
            audio_name,
            spectrogram,
        )
    except RuntimeWarning as e:
        print(f"Error normalizing spectrogram for '{audio_name}': {e}")
        return spectrogram  # Return unnormalized spectrogram on error


def z_normalize_spectrogram(spectrogram, audio_name=""):
    """
    Normalize the spectrogram using z-normalization across the time axis.
    Normalization can help reduce the influence of volume differences.
    """
    log_mel_mean = np.mean(spectrogram, axis=0)
    log_mel_std = np.std(spectrogram, axis=0)

    # Z-normalization formula
    normalized_log_mel = (spectrogram - log_mel_mean) / (
        log_mel_std + np.spacing(1)
    )

    try:
        return handle_warnings(
            lambda s: normalized_log_mel, audio_name, spectrogram
        )
    except RuntimeWarning as e:
        print(f"Error normalizing spectrogram for '{audio_name}': {e}")
        return spectrogram  # Return unnormalized spectrogram on error


# Comparison Functions
# NOTE: Reverse the order of audios for Ez Generated Model as generations are smaller than originals
def compare_subsequence_dtw(log_mel_orig, log_mel_aug):
    """
    Perform a subsequence DTW comparison between two log-mel spectrograms.
    Compare the entire original spectrogram against subsequences of the
    augmented spectrogram.
    :param log_mel_orig: Log-mel spectrogram of the original audio.
    :param log_mel_aug: Log-mel spectrogram of the augmented audio.
    :return: Average DTW distance between the original and any subsequence of the augmented spectrogram.
    """
    dummy_var = log_mel_orig
    log_mel_orig = log_mel_aug
    log_mel_aug = dummy_var

    len_orig = log_mel_orig.shape[1]
    len_aug = log_mel_aug.shape[1]

    if len_orig > len_aug:
        raise ValueError(
            "Augmented spectrogram is longer than original spectrogram. Subsequence DTW cannot be applied."
        )

    dtw_distances = []  # Store DTW distances

    # Sliding window over the augmented spectrogram
    for i in range(len_aug - len_orig + 1):
        # Extract a window from the augmented spectrogram that matches the length of the original
        windowed_aug = log_mel_aug[:, i : i + len_orig]

        # Compute the DTW distance for this windowed section
        distance, _ = librosa.sequence.dtw(
            X=log_mel_orig.T, Y=windowed_aug.T, metric="euclidean"
        )
        dtw_distance = distance[-1, -1]  # Use the final distance in the matrix
        # as it's the max distance amongst the original sequence and aug subsequence

        # Keep track of the DTW distance
        dtw_distances.append(dtw_distance)

    # Normalize DTW distances
    max_dtw_distance = np.max(dtw_distances)
    min_dtw_distance = np.min(dtw_distances)

    if max_dtw_distance == min_dtw_distance:
        # Normalize distances to the range [0, 1]
        normalized_dtw = (dtw_distances - min_dtw_distance) / (
            np.spacing(1) + (max_dtw_distance - min_dtw_distance)
        )
    else:
        normalized_dtw = (dtw_distances - min_dtw_distance) / (
            (max_dtw_distance - min_dtw_distance)
        )

    return np.mean(normalized_dtw)  # 0 similar, 1 is dissimilar


def compare_sliding_window_cross_correlation(log_mel_orig, log_mel_aug):
    """
    Compute the Sliding Window Cross-Correlation between two log-mel spectrograms.

    :param log_mel_orig: Log-mel spectrogram of the original audio.
    :param log_mel_aug: Log-mel spectrogram of the augmented audio.
    :return: Average cross-correlation score.
    """
    dummy_var = log_mel_orig
    log_mel_orig = log_mel_aug
    log_mel_aug = dummy_var

    if log_mel_aug.shape[1] < log_mel_orig.shape[1]:
        raise ValueError(
            "Original spectrogram must be at least as long as the Augmented."
        )

    corrs = []
    # Iterate over the augmented spectrogram with a sliding window of size of original log_mel
    for i in range(log_mel_aug.shape[1] - log_mel_orig.shape[1] + 1):
        # Extract a window from the augmented spectrogram
        windowed_aug = log_mel_aug[:, i : i + log_mel_orig.shape[1]]

        # Compute the correlation coefficient
        correlation = np.corrcoef(
            log_mel_orig.flatten(), windowed_aug.flatten()
        )[
            0, 1
        ]  # Range: -1 to 1
        corrs.append(correlation)  # Append correlation to the list

    # Return the average correlation score (inverted for distance)
    return 1 - np.mean(
        corrs
    )  # (0 is similar, 1 is dissimilar, 2 is negatively correlated)


def compare_model_embeddings(log_mel_orig, log_mel_aug):

    orig_audio_vec = torch.unsqueeze(
        torch.as_tensor(log_mel_orig).to("cuda"), dim=0
    )

    aug_audio_vec = torch.unsqueeze(
        torch.as_tensor(log_mel_aug).to("cuda"), dim=0
    )
    orig_audio_embed = F.normalize(
        cnn14_encoder(orig_audio_vec), p=2.0, dim=-1
    )
    aug_audio_embed = F.normalize(cnn14_encoder(aug_audio_vec), p=2.0, dim=-1)

    similarity = F.cosine_similarity(orig_audio_embed, aug_audio_embed, dim=1)

    # Return dissimilarity score (1 - similarity)
    return 1 - similarity.item()


## Audio Processing Functions
def process_audio_file(audio_path, input_fn, method):
    """
    Process a single audio file using a custom processing function.

    :param audio_path: Path to the audio file.
    :param input_fn: Function to extract features from audio.
    :return: Extracted features.
    """
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception as e:
        print(f"Error loading audio file '{audio_path}': {e}")
        return None

    features = input_fn(y, sr, name=audio_path, method=method)
    if features is None:
        print(f"Error processing audio file '{audio_path}'")
    return features


def compare_audio_files(
    original_audio_path, augmented_audio_path, comparison_fn, input_fn, method
):
    """
    Compare two audio files using a custom comparison function.

    :param original_audio_path: Path to the original audio file.
    :param augmented_audio_path: Path to the augmented audio file.
    :param comparison_fn: Function to compare the features of two audio files.
    :param input_fn: Function to extract features from an audio file.
    :return: Comparison score.
    """
    # Process the audio files and get features (e.g., log-mel spectrograms)
    original_features = min_max_normalize_spectrogram(
        process_audio_file(original_audio_path, input_fn, method),
        audio_name=original_audio_path,
    )
    augmented_features = min_max_normalize_spectrogram(
        process_audio_file(augmented_audio_path, input_fn, method),
        audio_name=augmented_audio_path,
    )

    # Skip comparison if either feature extraction failed
    if original_features is None or augmented_features is None:
        return None

    # Compare the features and compute similarity score
    similarity_score = comparison_fn(original_features, augmented_features)
    return similarity_score


def compare_audio_folders_for_development(
    original_folder,
    augmented_folder,
    fid_data,
    input_fn,
    comparison_fn,
    cap_num,
    method,
    NUM=None,
):
    """
    Compare the audio files in the development split and store the comparison scores in a CSV file.
    """
    split = "development"
    fid2fname = fid_data[split]

    # Limit the number of audio files if NUM is specified
    file_items = list(fid2fname.items())[:NUM] if NUM else fid2fname.items()

    # List to hold the comparison results
    results = []

    for fid, fname in tqdm(
        file_items, desc=f"Comparing {split} audios", total=len(file_items)
    ):
        original_audio_path = os.path.join(original_folder, split, fname)

        # Construct augmented audio path with "_cap_{cap_num}.wav" suffix
        augmented_audio_filename = (
            f"{os.path.splitext(fname)[0]}_cap_{cap_num}.wav"
        )
        augmented_audio_path = os.path.join(
            augmented_folder, split, augmented_audio_filename
        )

        # Skip comparison if the corresponding augmented file doesn't exist
        if not os.path.exists(original_audio_path) or not os.path.exists(
            augmented_audio_path
        ):
            continue

        # Compare the audio files
        similarity_score = compare_audio_files(
            original_audio_path,
            augmented_audio_path,
            comparison_fn,
            input_fn,
            method,
        )

        # Append result to list if similarity_score is not None
        if similarity_score is not None:
            results.append(
                {
                    "filename": fname,
                    "fid": fid,
                    "similarity_score": similarity_score,
                }
            )
        else:
            print(f"None similarity score for {original_audio_path}")

    return results


def process_fid_chunks(
    original_folder,
    augmented_folder,
    fid_file_path,
    output_csv_path,
    input_fn,
    comparison_fn,
    cap_num,
    num_chunks,
    method,
    NUM=None,
):
    """
    Process audio files in parallel by splitting the fid data into chunks.
    Only process the first NUM files if specified.
    """
    # Load the fid file
    with open(fid_file_path, "rb") as f:
        fid_data = pickle.load(f)
        audio_fid2fname = fid_data["audio_fid2fname"]

    split = "development"
    fid2fname = audio_fid2fname[split]
    file_items = list(fid2fname.items())  # All audio files

    # Limit the number of audio files if NUM is specified
    if NUM is not None:
        file_items = file_items[:NUM]  # Select the first NUM items

    # Split the file_items into chunks
    chunk_size = len(file_items) // num_chunks
    chunks = [
        file_items[i : i + chunk_size]
        for i in range(0, len(file_items), chunk_size)
    ]

    results = []  # List to store results from all processes

    with ProcessPoolExecutor() as executor:
        # Submit each chunk for processing
        futures = {
            executor.submit(
                compare_audio_folders_for_development,
                original_folder,
                augmented_folder,
                {split: dict(chunk)},
                input_fn,
                comparison_fn,
                cap_num,
                method,
            ): chunk
            for chunk in chunks
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing chunks"
        ):
            result = future.result()
            results.extend(result)  # Combine results from each chunk

    # Convert the results to a pandas DataFrame
    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by="filename")
    # Save the DataFrame to a CSV file
    df_sorted.to_csv(output_csv_path, index=False)
    print(f"Comparison results saved to {output_csv_path}")


def process_audio_files(
    original_folder,
    augmented_folder,
    fid_file_path,
    output_csv_path,
    input_fn,
    comparison_fn,
    cap_num,
    method,
    NUM=None,
):
    """
    Process a limited number of audio files directly without multiprocessing.
    """
    # Load the fid file
    with open(fid_file_path, "rb") as f:
        fid_data = pickle.load(f)
        audio_fid2fname = fid_data["audio_fid2fname"]

    split = "development"
    fid2fname = audio_fid2fname[split]
    file_items = list(fid2fname.items())[:NUM]  # Limit to NUM items

    # Compare audio files
    results = compare_audio_folders_for_development(
        original_folder,
        augmented_folder,
        {split: dict(file_items)},
        input_fn,
        comparison_fn,
        cap_num,
        method,
        NUM,
    )

    # Convert the results to a pandas DataFrame
    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by="filename")
    # Save the DataFrame to a CSV file
    df_sorted.to_csv(output_csv_path, index=False)
    print(f"Comparison results saved to {output_csv_path}")


@click.command()
@click.option(
    "--cap_num", default=None, type=int, help="Caption Column to Process"
)
@click.option(
    "--method",
    type=click.Choice(["dtw", "wcc", "model"], case_sensitive=False),
    help="Comparison method to use.",
)
def main(cap_num, method):
    original_folder = "./data/Clotho"  # Path to original audio folder
    fid_file_path = "./data/Clotho/audio_info.pkl"  # Path to fid file
    input_fn = log_mel_spectrogram  # Function to extract features

    augmented_folder = (
        f"./data/ez_clotho_caption_{cap_num}"  # Path to augmented audio folder
    )

    output_csv_path = f"./temp/ezmodel_original_vs_cap_{cap_num}_generated_audio_via_{method}.csv"

    if method == "wcc":
        comparison_fn = compare_sliding_window_cross_correlation
        process_fid_chunks(
            original_folder,
            augmented_folder,
            fid_file_path,
            output_csv_path,
            input_fn,
            comparison_fn,
            cap_num,
            num_chunks=24,
            method=method,
            NUM=None,
        )
    elif method == "dtw":
        comparison_fn = compare_subsequence_dtw
        process_fid_chunks(
            original_folder,
            augmented_folder,
            fid_file_path,
            output_csv_path,
            input_fn,
            comparison_fn,
            cap_num,
            num_chunks=24,
            method=method,
            NUM=None,
        )
    elif method == "model":
        comparison_fn = compare_model_embeddings
        process_audio_files(
            original_folder,
            augmented_folder,
            fid_file_path,
            output_csv_path,
            input_fn,
            comparison_fn,
            cap_num,
            method=method,
            NUM=None,
        )


if __name__ == "__main__":
    main()
