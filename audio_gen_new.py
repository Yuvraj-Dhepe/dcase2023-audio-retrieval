
import os
import pandas as pd
import typing as tp
from pathlib import Path
import torch
from audiocraft.models import AudioGen
from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.audio import audio_write
import time
from tqdm import tqdm  # Import tqdm for progress tracking
import shutil  # Import shutil for file operations

torch.manual_seed(24)
class FileCleaner:
    def __init__(self, file_lifetime: float = 3600):
        self.file_lifetime = file_lifetime
        self.files = []

    def add(self, path: tp.Union[str, Path]):
        self._cleanup()
        self.files.append((time.time(), Path(path)))

    def _cleanup(self):
        now = time.time()
        self.files = [(t, p) for t, p in self.files if now - t <= self.file_lifetime]
        for _, path in self.files:
            if path.exists():
                path.unlink()

    def cleanup_all(self):
        for _, path in self.files:
            if path.exists():
                path.unlink()
        self.files.clear()

def load_model() -> AudioGen:
    return AudioGen.get_pretrained('facebook/audiogen-medium')

def set_generation_params(model: AudioGen):
    model.set_generation_params(
        use_sampling=True,
        top_k=250, top_p=0.0, temperature=1.0,
        cfg_coef=3.0, extend_stride=model.max_duration - 1, duration=30
    )

def generate_audio(caption: str, model: AudioGen):
    start_time = time.time()
    outputs = model.generate(descriptions=[caption], progress=False)
    duration = time.time() - start_time
    # print(f"Time taken to generate audio: {duration:.2f} seconds")
    return convert_audio(outputs, from_rate=16000, to_rate=44100, to_channels=1)

def process_csv(csv_file_path: str, output_folder: Path, model: AudioGen,caption_col_num:int):
    file_cleaner = FileCleaner()
    output_folder.mkdir(exist_ok=True)
    df = pd.read_csv(csv_file_path)

    folder_count = 1  # Start with folder_1
    current_folder = output_folder / str(folder_count)
    current_folder.mkdir(exist_ok=True)
    file_count = 0

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Captions"):
        file_name = row['file_name']
        caption = row[f'caption_{caption_col_num}']
        outputs = generate_audio(caption, model)
        save_audio_to_folder(outputs, file_name, current_folder, file_cleaner, caption_col_num)

        file_count += 1
        if file_count >= 500:
            file_count = 0
            folder_count += 1
            current_folder = output_folder / str(folder_count)
            current_folder.mkdir(exist_ok=True)

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    file_cleaner.cleanup_all()


def save_audio_to_folder(outputs, file_name: str, output_folder: Path, file_cleaner: FileCleaner, caption_col_num):
    # Strip the .wav extension if it exists
    file_base_name = Path(file_name).stem
    output_file_name = output_folder / f"{file_base_name}_cap_{caption_col_num}"
    # output_file_name,_ =  os.path.splitext(output_file_name)
    # print(output_file_name)
    # Directly write the audio file to the output folder
    audio_write(
        output_file_name, outputs[0], 44100, strategy="loudness",
        loudness_headroom_db=14, loudness_compressor=True
    )

    file_cleaner.add(output_file_name)
    # print(f"Generated audio saved as: {output_file_name}")

if __name__ == "__main__":
    caption_col_num = 5
    caption_col_folder = f"Clotho_caption_{caption_col_num}"
    split = 'development'
    model = load_model()
    set_generation_params(model)
    # remove_extra_wav_suffix(Path("./data/Clotho/development_gen"))

    path = f"./data/{caption_col_folder}/{split}"
    os.makedirs(path, exist_ok=True)
    process_csv('./curated_clotho_captions/clotho_captions_development.csv',
                Path(path), model,caption_col_num)