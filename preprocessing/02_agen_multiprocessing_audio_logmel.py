import glob
import os
import pickle
import h5py
import librosa
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm  # Import tqdm


def log_mel_spectrogram(
    y,
    sample_rate=44100,
    window_length_secs=0.025,
    hop_length_secs=0.010,
    num_mels=128,
    log_offset=0.0,
):
    """
    Convert waveform to a log magnitude mel-frequency spectrogram.

    :param y: 1D np.array of waveform data.
    :param sample_rate: The sampling rate of data.
    :param window_length_secs: Duration of each window to analyze.
    :param hop_length_secs: Advance between successive analysis windows.
    :param num_mels: Number of Mel bands.
    :param log_offset: Add this to values when taking log to avoid -Infs.
    :return: np.array of log-mel spectrogram.
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

    return np.log(mel_spectrogram + log_offset)


def process_audio(fpath, fname2fid):
    """
    Process a single audio file to extract log-mel spectrogram.

    :param fpath: Path to the audio file.
    :param fname2fid: Dictionary mapping filenames to file IDs (fids).
    :return: Tuple of (fid, log-mel spectrogram).
    """
    fname = os.path.basename(fpath)
    fid = fname2fid[fname]

    y, sr = librosa.load(fpath, sr=None, mono=True)
    assert len(y.shape) == 1, "Audio is not mono!"

    log_mel = log_mel_spectrogram(
        y,
        sr,
        window_length_secs=0.040,
        hop_length_secs=0.020,
        num_mels=64,
        log_offset=np.spacing(1),
    )
    # TODO: remove stacking and once check the baseline scores are they impacted or not
    # Stacking is useless as the log_mel is 2d, and vstack returns the same array.
    return fid, np.vstack(log_mel).transpose()


def extract_log_mel_spectrograms(dataset_dirs, audio_splits, audio_fid2fname):
    """
    Extract log-mel spectrograms for all audio files in the specified datasets and splits.

    :param dataset_dirs: List of directories containing audio data.
    :param audio_splits: List of audio splits (e.g., "development", "validation").
    :param audio_fid2fname: Dictionary mapping audio splits to fid-to-filename mappings.
    """
    for split in audio_splits:
        if split == "development":
            used_dataset_dirs = dataset_dirs  # Use all directories
        else:
            used_dataset_dirs = [
                dataset_dirs[0]
            ]  # Use only the first directory

        fid2fname = audio_fid2fname[split]
        fname2fid = {fid2fname[fid]: fid for fid in fid2fname}

        audio_logmel = os.path.join(
            dataset_dirs[-1], f"{split}_audio_logmels.hdf5"
        )

        with h5py.File(audio_logmel, "w") as stream:

            all_wav_files = []
            for dataset_dir in used_dataset_dirs:
                audio_dir = os.path.join(dataset_dir, split)
                all_wav_files.extend(glob.glob(r"{}/*.wav".format(audio_dir)))

            with Pool(processes=os.cpu_count()) as pool:
                # Use a partial function to pass fname2fid to process_audio
                from functools import partial

                process_audio_with_fname2fid = partial(
                    process_audio, fname2fid=fname2fid
                )

                results = list(
                    tqdm(
                        pool.imap(process_audio_with_fname2fid, all_wav_files),
                        total=len(all_wav_files),
                        desc=f"Processing audio files in {split}",
                    )
                )

            for fid, log_mel in results:
                stream[fid] = log_mel

        print("Save", audio_logmel)
        print("=" * 90)


# Main execution
if __name__ == "__main__":
    global_params = {
        "dataset_dirs": [
            "./data/Clotho",
            "./data/Clotho_caption_5",
        ],  # Modified
        "audio_splits": ["development", "validation", "evaluation"],
    }

    # Load audio info
    audio_info = os.path.join(
        global_params["dataset_dirs"][-1], "audio_info.pkl"
    )
    with open(audio_info, "rb") as store:
        audio_fid2fname = pickle.load(store)["audio_fid2fname"]

    # Extract log mel for splits
    extract_log_mel_spectrograms(
        global_params["dataset_dirs"],
        global_params["audio_splits"],
        audio_fid2fname,
    )
