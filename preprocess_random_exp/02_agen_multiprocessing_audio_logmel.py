import glob
import os
import pickle
import h5py
import librosa
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm  # Import tqdm for progress tracking
from functools import partial

def log_mel_spectrogram(y, sample_rate=44100, window_length_secs=0.025, hop_length_secs=0.010, num_mels=128, log_offset=0.0):
    """
    Convert a waveform into a log-magnitude mel-frequency spectrogram.

    :param y: Audio time series (waveform).
    :param sample_rate: Sampling rate of the audio.
    :param window_length_secs: Length of each FFT window in seconds.
    :param hop_length_secs: Hop length between windows in seconds.
    :param num_mels: Number of Mel bands to generate.
    :param log_offset: Offset added to avoid taking log of zero.
    :return: Log-magnitude mel-frequency spectrogram.
    """
    # Convert seconds to samples
    window_length = int(round(sample_rate * window_length_secs))
    hop_length = int(round(sample_rate * hop_length_secs))

    # Compute FFT length, rounded up to the nearest power of 2
    fft_length = 2 ** int(np.ceil(np.log(window_length) / np.log(2.0)))

    # Compute mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sample_rate, n_fft=fft_length, hop_length=hop_length,
        win_length=window_length, n_mels=num_mels
    )

    # Return the log of the mel spectrogram to make it log-magnitude
    return np.log(mel_spectrogram + log_offset)

def process_audio(fpath, fname2fid):
    """
    Process a single audio file to extract its log-mel spectrogram.

    :param fpath: File path of the audio file.
    :param fname2fid: Dictionary mapping filenames to file IDs (fids).
    :return: Tuple of file ID and log-mel spectrogram.
    """
    fname = os.path.basename(fpath)  # Get the file name from the file path
    fid = fname2fid.get(fname)  # Retrieve the corresponding file ID (fid)

    if fid is None:
        # If the file is not found in the mapping, return None
        print(f"File {fname} not found in the given mapping.")
        return None

    # Load the audio file using librosa
    y, sr = librosa.load(fpath, sr=None, mono=True)  # sr=None keeps the original sampling rate
    assert len(y.shape) == 1, "Audio is not mono!"  # Ensure the audio is mono

    # Compute log-mel spectrogram with specific parameters
    log_mel = log_mel_spectrogram(
        y, sr, window_length_secs=0.040, hop_length_secs=0.020,
        num_mels=64, log_offset=np.spacing(1)
    )

    # Return the file ID and the log-mel spectrogram transposed for correct shape
    # TODO: remove stacking and once check the baseline scores are they impacted or not
    return fid, np.vstack(log_mel).transpose()

def extract_log_mel_spectrograms(dataset_dirs, audio_splits, audio_fid2fname, output_dir):
    """
    Extract log-mel spectrograms for selected audio files in the given dataset splits.

    :param dataset_dirs: List of directories containing the datasets.
    :param audio_splits: List of dataset splits (e.g., "development", "validation", "evaluation").
    :param audio_fid2fname: Dictionary mapping file IDs to filenames for each split.
    :param output_dir: Directory to save the extracted log-mel spectrograms.
    """
    for split in audio_splits:
        if split == "development":
            # Use all directories for the 'development' split
            used_dataset_dirs = dataset_dirs
        else:
            # Use only the first directory for other splits (validation, evaluation)
            used_dataset_dirs = [dataset_dirs[0]]

        # Get the fid-to-filename mapping for the current split
        fid2fname = audio_fid2fname[split]
        fname2fid = {fname: fid for fid, fname in fid2fname.items()}  # Reverse the mapping to get fname -> fid

        # Output HDF5 file path to store log-mel spectrograms
        audio_logmel = os.path.join(output_dir, f"{split}_audio_logmels.hdf5")

        # Open the HDF5 file for writing
        with h5py.File(audio_logmel, "w") as stream:

            # Collect all audio file paths in the used dataset directories
            all_wav_files = []
            for dataset_dir in used_dataset_dirs:
                audio_dir = os.path.join(dataset_dir, split)
                all_wav_files.extend(glob.glob(r"{}/*.wav".format(audio_dir)))

            # Filter the files to only process the ones that are selected (exist in audio_info.pkl)
            selected_wav_files = [f for f in all_wav_files if os.path.basename(f) in fname2fid]

            if not selected_wav_files:
                print(f"No files selected for processing in {split}.")
                continue

            # Process the selected audio files in parallel using multiprocessing Pool
            with Pool(processes=os.cpu_count()) as pool:
                # Use partial to pass fname2fid as an additional argument to process_audio
                process_audio_with_fname2fid = partial(process_audio, fname2fid=fname2fid)

                # Use tqdm to track the progress of audio file processing
                results = list(tqdm(pool.imap(process_audio_with_fname2fid, selected_wav_files),
                                    total=len(selected_wav_files),
                                    desc=f"Processing selected audio files in {split}"))

            # Save the processed log-mel spectrograms to the HDF5 file
            for result in results:
                if result is not None:
                    fid, log_mel = result
                    stream[fid] = log_mel  # Save the log-mel spectrogram under its fid

        print(f"Saved log-mel spectrograms to {audio_logmel}")
        print('=' * 90)

def load_audio_info(audio_info_path):
    """
    Load audio file information from a pickle file.

    :param audio_info_path: Path to the pickle file containing audio info.
    :return: Audio file information loaded from the pickle file.
    """
    with open(audio_info_path, "rb") as store:
        return pickle.load(store)

# Main execution
if __name__ == "__main__":
    global_params = {
        # List of dataset directories (for both original and synthetic data)
        "dataset_dirs": [
            "./data/Clotho", "./data/Clotho_caption_1", "./data/Clotho_caption_2",
            "./data/Clotho_caption_3", "./data/Clotho_caption_4", "./data/Clotho_caption_5"
        ],
        # Audio splits to process
        "audio_splits": ["development", "validation", "evaluation"]
    }

    # Directory to save the output
    output_dir = './data/exp_5'

    # Path to the pickle file containing audio file info
    audio_info_path = os.path.join(output_dir, "audio_info.pkl")

    # Load the audio file information from the pickle file
    audio_data = load_audio_info(audio_info_path)

    # Extract the log-mel spectrograms for the selected audio files
    extract_log_mel_spectrograms(global_params["dataset_dirs"], global_params["audio_splits"], audio_data["audio_fid2fname"], output_dir)
