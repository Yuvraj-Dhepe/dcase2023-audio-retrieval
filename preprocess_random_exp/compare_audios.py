import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
from scipy.spatial.distance import euclidean
import warnings
import matplotlib.pyplot as plt
import librosa.display

# Function to handle warnings globally
def handle_warnings(func, name = "",*args, **kwargs):
    """Handle warnings globally for any function."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # Always trigger warnings
        result = func(*args, **kwargs)  # Call the function
        if w:
            for warning in w:
                print(f"Warning for {name}: {warning.message}")  # Print warning message
    return result

def visualize_log_mel_spectrogram(log_mel, sample_rate=44100, hop_length_secs=0.010, title='Log-Mel Spectrogram'):
    """
    Visualize a log-mel spectrogram using librosa's specshow and matplotlib.

    :param log_mel: Log-mel spectrogram to visualize.
    :param sample_rate: Sample rate of the audio.
    :param hop_length_secs: Hop length in seconds for the spectrogram.
    :param title: Title of the plot.
    """
    hop_length = int(round(sample_rate * hop_length_secs))  # Calculate the hop length in samples

    plt.figure(figsize=(10, 6))
    # Display the log-mel spectrogram using librosa's specshow
    librosa.display.specshow(log_mel, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='viridis')

    # Add color bar and title
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def log_mel_spectrogram(y, sample_rate=44100, window_length_secs=0.025,
                        hop_length_secs=0.010, num_mels=128, log_offset=1e-9, name=''):
    """
    Convert waveform to a log magnitude mel-frequency spectrogram.
    """
    window_length = int(round(sample_rate * window_length_secs))
    hop_length = int(round(sample_rate * hop_length_secs))
    fft_length = 2 ** int(np.ceil(np.log(window_length) / np.log(2.0)))

    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sample_rate, n_fft=fft_length, hop_length=hop_length,
        win_length=window_length, n_mels=num_mels
    )
      # Ensure no negative or zero values before applying log
    mel_spectrogram = np.maximum(mel_spectrogram, log_offset)

    try:
        log_mel = handle_warnings(np.log,name,mel_spectrogram)  # Log mel spectrogram with warning handling
    except ValueError as e:
        print(f"Error calculating log-mel spectrogram: {e}")
        return None

    return log_mel

def compare_log_mel_spectrograms(log_mel_orig, log_mel_aug):
    """
    Compare two log-mel spectrograms using Euclidean distance.
    """
    min_time_steps = min(log_mel_orig.shape[1], log_mel_aug.shape[1])
    log_mel_orig = librosa.util.fix_length(log_mel_orig, size=min_time_steps, axis=1)
    log_mel_aug = librosa.util.fix_length(log_mel_aug, size=min_time_steps, axis=1)

    # Euclidean distance between the log-mel spectrograms
    similarity_score = np.linalg.norm(log_mel_orig - log_mel_aug)
    return similarity_score



def compare_log_mel_spectrograms_cosine(log_mel_orig, log_mel_aug):
    """Compare two log-mel spectrograms using cosine similarity."""
    """If Euclidean distance is producing unexpectedly high scores, consider using a different similarity metric, like cosine similarity, which is less sensitive to absolute differences in magnitude and focuses more on the shape of the spectrograms."""
    min_time_steps = min(log_mel_orig.shape[1], log_mel_aug.shape[1])
    log_mel_orig = librosa.util.fix_length(log_mel_orig, size=min_time_steps, axis=1)
    log_mel_aug = librosa.util.fix_length(log_mel_aug, size=min_time_steps, axis=1)

    similarity_score = cosine_similarity(log_mel_orig.T, log_mel_aug.T)[0][0]  # Get the cosine similarity score
    return 1 - similarity_score  # Convert to distance (0 is similar, 1 is dissimilar)


def compare_dtw_distance(log_mel_orig, log_mel_aug):
    """
    # https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3S2_DTWbasic.html
    Compute the Dynamic Time Warping (DTW) distance between two log-mel spectrograms.

    :param log_mel_orig: Log-mel spectrogram of the original audio.
    :param log_mel_aug: Log-mel spectrogram of the augmented audio.
    :return: DTW distance.
    """
    distance, _ = librosa.sequence.dtw(X=log_mel_orig.T, Y=log_mel_aug.T, metric='euclidean')
    return distance[-1, -1]  # Return the last element of the distance matrix

def compare_sliding_window_cross_correlation(log_mel_orig, log_mel_aug):
    """
    Compute the Sliding Window Cross-Correlation between two log-mel spectrograms.

    :param log_mel_orig: Log-mel spectrogram of the original audio.
    :param log_mel_aug: Log-mel spectrogram of the augmented audio.
    :return: Average cross-correlation score.
    """
    if log_mel_aug.shape[1] < log_mel_orig.shape[1]:
        raise ValueError("Augmented spectrogram must be at least as long as the original.")

    corrs = []
    # Iterate over the augmented spectrogram with a sliding window of size of original log_mel
    for i in range(log_mel_aug.shape[1] - log_mel_orig.shape[1] + 1):
        # Extract a window from the augmented spectrogram
        windowed_aug = log_mel_aug[:, i:i + log_mel_orig.shape[1]]

        # Compute the correlation coefficient
        correlation = np.corrcoef(log_mel_orig.flatten(), windowed_aug.flatten())[0, 1]  # Range: -1 to 1
        corrs.append(correlation)  # Append correlation to the list

    # Return the average correlation score (inverted for distance)
    return 1 - np.mean(corrs)  # (0 is similar, 1 is dissimilar, 2 is negatively correlated)

def normalize_spectrogram(spectrogram, audio_name =''):
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
        return handle_warnings(lambda s: (s - min_val) / (max_val - min_val),audio_name,spectrogram)
    except RuntimeWarning as e:
        print(f"Error normalizing spectrogram for '{audio_name}': {e}")
        return spectrogram  # Return unnormalized spectrogram on error

def process_audio_file(audio_path, process_fn):
    """
    Process a single audio file using a custom processing function.

    :param audio_path: Path to the audio file.
    :param process_fn: Function to extract features from audio.
    :return: Extracted features.
    """
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception as e:
        print(f"Error loading audio file '{audio_path}': {e}")
        return None

    features = process_fn(y, sr, name=audio_path)
    if features is None:
        print(f"Error processing audio file '{audio_path}'")
    return features


def compare_audio_files(original_audio_path, augmented_audio_path, comparison_fn, process_fn):
    """
    Compare two audio files using a custom comparison function.

    :param original_audio_path: Path to the original audio file.
    :param augmented_audio_path: Path to the augmented audio file.
    :param comparison_fn: Function to compare the features of two audio files.
    :param process_fn: Function to extract features from an audio file.
    :return: Comparison score.
    """
    # Process the audio files and get features (e.g., log-mel spectrograms)
    original_features = normalize_spectrogram(process_audio_file(original_audio_path, process_fn), audio_name=original_audio_path)
    augmented_features = normalize_spectrogram(process_audio_file(augmented_audio_path, process_fn), audio_name=augmented_audio_path)

    # Skip comparison if either feature extraction failed
    if original_features is None or augmented_features is None:
        return None

    # Compare the features and compute similarity score
    similarity_score = comparison_fn(original_features, augmented_features)
    return similarity_score


def compare_audio_folders_for_development(original_folder, augmented_folder, fid_file_path, output_csv_path, process_fn, comparison_fn, cap_num, NUM=None):
    """
    Compare the audio files in the development split and store the comparison scores in a CSV file.
    """
    # Load the fid file
    with open(fid_file_path, 'rb') as f:
        fid_data = pickle.load(f)
        audio_fid2fname = fid_data["audio_fid2fname"]

    split = "development"
    fid2fname = audio_fid2fname[split]

    # Limit the number of audio files if NUM is specified
    file_items = list(fid2fname.items())[:NUM] if NUM else fid2fname.items()

    # List to hold the comparison results
    results = []

    for fid, fname in tqdm(file_items, desc=f"Comparing {split} audios", total=len(file_items)):
        original_audio_path = os.path.join(original_folder, split, fname)

        # Construct augmented audio path with "_cap_{cap_num}.wav" suffix
        augmented_audio_filename = f"{os.path.splitext(fname)[0]}_cap_{cap_num}.wav"
        augmented_audio_path = os.path.join(augmented_folder, split, augmented_audio_filename)

        # Skip comparison if the corresponding augmented file doesn't exist
        if not os.path.exists(original_audio_path) or not os.path.exists(augmented_audio_path):
            continue

        # Compare the audio files
        similarity_score = compare_audio_files(original_audio_path, augmented_audio_path, comparison_fn, process_fn)

        # Append result to list
         # Append result to list if similarity_score is not None
        if similarity_score is not None:
            results.append({"filename": fname, "fid": fid, "similarity_score": similarity_score})
        else:
            print(f"None similarity score for {original_audio_path}")

    # Convert the results to a pandas DataFrame
    df = pd.DataFrame(results)

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv_path, index=False)
    print(f"Comparison results saved to {output_csv_path}")


# Main function
if __name__ == "__main__":
    method = 'wcc'
    i = 1

    original_folder = './data/Clotho'  # Path to original audio folder
    augmented_folder = f'./data/Clotho_caption_{i}'  # Path to augmented audio folder
    fid_file_path = 'data/Clotho/audio_info.pkl'  # Path to fid file will always be fixed, as we want to always compare only the original audios with the generated ones.

    output_csv_path = f'./temp/original_vs_cap_{i}_generated_audio_via_{method}.csv'  # Path to output comparison CSV file

    # Specify the method to use for processing and comparison
    process_fn = log_mel_spectrogram  # Function to extract features (log-mel spectrogram)
    comparison_fn = compare_sliding_window_cross_correlation  # Function to compare features

    NUM = None  # Set to None to compare all audios or specify a number to limit to the first K audios

    compare_audio_folders_for_development(original_folder, augmented_folder, fid_file_path, output_csv_path, process_fn, comparison_fn, i, NUM=NUM)



### Why Compare Log Mel Spectrogams
# Direct Match to Model Input: Since you're using log-mel-spectrograms during training, comparing the same feature ensures that the similarity measure is relevant to the model's input.
# Frequency Representation: Log-mel-spectrograms capture both temporal and spectral information, which is key in understanding how similar the augmentations are to the original data.
# Computationally Efficient: This method is more efficient than something like DTW for large datasets, as it works with the spectrogram format directly.
# This method will give you a numerical score that represents the similarity between the original and augmented audio in the feature space your model is using. You can adjust the distance metric or normalization as needed, but Euclidean distance or Cosine similarity on the log-mel-spectrogram is generally effective for such tasks.