import os
from pathlib import Path
import librosa
import typing as tp
import torch
from tqdm import tqdm
import pandas as pd
import soundfile as sf
import time
from api.ezaudio import EzAudio


class FileCleaner:
    def __init__(self, file_lifetime: float = 3600):
        self.file_lifetime = file_lifetime
        self.files = []

    def add(self, path: tp.Union[str, Path]):
        self._cleanup()
        self.files.append((time.time(), Path(path)))

    def _cleanup(self):
        now = time.time()
        self.files = [
            (t, p) for t, p in self.files if now - t <= self.file_lifetime
        ]
        for _, path in self.files:
            if path.exists():
                path.unlink()

    def cleanup_all(self):
        for _, path in self.files:
            if path.exists():
                path.unlink()
        self.files.clear()


def resample_audio(input_file, output_file, target_sr):
    """
    Resamples an audio file to the specified target sampling rate.

    Parameters:
        input_file (str): Path to the input audio file.
        output_file (str): Path to save the resampled audio file.
        target_sr (int): Target sampling rate (e.g., 44100).

    Returns:
        None
    """
    # Load the audio at its original sampling rate
    audio, sr = librosa.load(input_file, sr=None)  # Preserve the original SR
    # print(f"Original Sample Rate: {sr} Hz")

    # Resample to the target sampling rate
    resampled_audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    # Save the resampled audio
    sf.write(output_file, resampled_audio, target_sr)
    # print(f"Resampled audio saved to '{output_file}' with SR={target_sr} Hz.")


def save_audio_to_folder(
    audio,
    sr,
    file_name: str,
    output_folder: Path,
    file_cleaner: FileCleaner,
    caption_col_num,
):
    # Strip the .wav extension if it exists
    file_base_name = Path(file_name).stem
    output_file_name = (
        output_folder / f"{file_base_name}_cap_{caption_col_num}"
    )
    # output_file_name,_ =  os.path.splitext(output_file_name)
    # print(output_file_name)
    # Directly write the audio file to the output folder
    sf.write("a1.wav", audio, sr)

    # Resample audio before saving
    resample_audio("a1.wav", f"{output_file_name}.wav", 44100)

    file_cleaner.add(output_file_name)


def process_csv(
    csv_file_path: str,
    output_folder: Path,
    caption_col_num: int,
):
    ezaudio = load_model()
    file_cleaner = FileCleaner()
    output_folder.mkdir(exist_ok=True)
    df = pd.read_csv(csv_file_path)

    folder_count = 1  # Start with folder_1
    current_folder = output_folder / str(folder_count)
    current_folder.mkdir(exist_ok=True)
    file_count = 0

    for _, row in tqdm(
        df.iterrows(), total=df.shape[0], desc="Processing Captions"
    ):
        file_name = row["file_name"]
        caption = row[f"caption_{caption_col_num}"]
        sr, audio = ezaudio.generate_audio(
            caption,
            length=10,
            guidance_scale=5,
            guidance_rescale=0.75,
            ddim_steps=200,
            random_seed=42,
        )

        save_audio_to_folder(
            audio, sr, file_name, current_folder, file_cleaner, caption_col_num
        )

        file_count += 1
        if file_count >= 500:
            file_count = 0
            folder_count += 1
            current_folder = output_folder / str(folder_count)
            current_folder.mkdir(exist_ok=True)

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    file_cleaner.cleanup_all()


def load_model() -> EzAudio:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ezaudio = EzAudio(model_name="s3_xl", device=device)
    return ezaudio


if __name__ == "__main__":

    split = "development"
    # remove_extra_wav_suffix(Path("./data/Clotho/development_gen"))
    caption_col_num = 5
    caption_col_folder = f"ez_clotho_caption_{caption_col_num}"
    path = f"/home/yuvidh/work/Projects/Audio_DL/dcase2023-audio-retrieval/data/{caption_col_folder}/{split}"
    csv = "/home/yuvidh/work/Projects/Audio_DL/dcase2023-audio-retrieval/curated_clotho_captions/clotho_captions_development.csv"
    os.makedirs(path, exist_ok=True)
    process_csv(
        csv,
        Path(path),
        caption_col_num,
    )

    # caption_col_num = 2
    # caption_col_folder = f"ez_clotho_caption_{caption_col_num}"
    # path = f"/home/yuvidh/work/Projects/Audio_DL/dcase2023-audio-retrieval/data/{caption_col_folder}/{split}"
    # os.makedirs(path, exist_ok=True)
    # process_csv(
    #     csv,
    #     Path(path),
    #     caption_col_num,
    # )

    # caption_col_num = 3
    # caption_col_folder = f"ez_clotho_caption_{caption_col_num}"
    # path = f"/home/yuvidh/work/Projects/Audio_DL/dcase2023-audio-retrieval/data/{caption_col_folder}/{split}"
    # os.makedirs(path, exist_ok=True)
    # process_csv(
    #     csv,
    #     Path(path),
    #     caption_col_num,
    # )

    # caption_col_num = 4
    # caption_col_folder = f"ez_clotho_caption_{caption_col_num}"
    # path = f"/home/yuvidh/work/Projects/Audio_DL/dcase2023-audio-retrieval/data/{caption_col_folder}/{split}"
    # os.makedirs(path, exist_ok=True)
    # process_csv(
    #     csv,
    #     Path(path),
    #     caption_col_num,
    # )

    # caption_col_num = 5
    # caption_col_folder = f"ez_clotho_caption_{caption_col_num}"
    # path = f"/home/yuvidh/work/Projects/Audio_DL/dcase2023-audio-retrieval/data/{caption_col_folder}/{split}"
    # os.makedirs(path, exist_ok=True)
    # process_csv(
    #     csv,
    #     Path(path),
    #     caption_col_num,
    # )
