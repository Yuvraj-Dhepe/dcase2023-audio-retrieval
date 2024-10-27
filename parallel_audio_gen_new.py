import dask
from dask.distributed import Client, LocalCluster
import pandas as pd
import torch
from audiocraft.models import AudioGen
from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.audio import audio_write
import time
from tqdm import tqdm  # Import tqdm for progress tracking
import typing as tp
from pathlib import Path

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


def load_model() -> AudioGen:
    return AudioGen.get_pretrained("facebook/audiogen-medium")


def set_generation_params(model: AudioGen):
    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        top_p=0.0,
        temperature=1.0,
        cfg_coef=3.0,
        extend_stride=model.max_duration - 1,
        duration=30,
    )


def generate_audio(caption: str, model: AudioGen):
    start_time = time.time()
    outputs = model.generate(descriptions=[caption], progress=False)
    duration = time.time() - start_time
    # print(f"Time taken to generate audio: {duration:.2f} seconds")
    return convert_audio(
        outputs, from_rate=16000, to_rate=44100, to_channels=1
    )


def save_audio(
    outputs, file_name: str, output_folder: Path, file_cleaner: FileCleaner
):
    # Strip the .wav extension if it exists
    file_base_name = Path(file_name).stem
    output_file_name = output_folder / f"{file_base_name}_cap_1"
    # output_file_name,_ =  os.path.splitext(output_file_name)
    # print(output_file_name)
    # Directly write the audio file to the output folder
    audio_write(
        output_file_name,
        outputs[0],
        44100,
        strategy="loudness",
        loudness_headroom_db=14,
        loudness_compressor=True,
    )

    file_cleaner.add(output_file_name)
    # print(f"Generated audio saved as: {output_file_name}")


def process_chunk(df_chunk: pd.DataFrame, output_folder: Path):
    model = load_model()
    set_generation_params(model)
    file_cleaner = FileCleaner()
    for _, row in tqdm(
        df_chunk.iterrows(),
        total=df_chunk.shape[0],
        desc="Processing Captions",
    ):
        file_name = row["file_name"]
        caption = row["caption_1"]
        outputs = generate_audio(caption, model)
        save_audio(outputs, file_name, output_folder, file_cleaner)

    file_cleaner.cleanup_all()
    del model  # unload the model


def process_csv(csv_file_path: str, output_folder: Path):
    df = pd.read_csv(csv_file_path)
    chunk_size = 10
    chunks = [
        df[i : i + chunk_size] for i in range(0, df.shape[0], chunk_size)
    ]
    output_folders = [output_folder / str(i + 1) for i in range(len(chunks))]
    for folder in output_folders:
        folder.mkdir(exist_ok=True)

    cluster = LocalCluster(
        n_workers=3,
        threads_per_worker=1,
        memory_limit="16GB",
        dashboard_address=":8797",
        resources={"GPU": 1},
    )
    client = Client(cluster)

    futures = []
    for chunk, folder in zip(chunks, output_folders):
        future = client.submit(
            process_chunk, chunk, folder, resources={"GPU": 1}
        )
        futures.append(future)

    results = client.gather(futures)

    client.close()
    cluster.close()


if __name__ == "__main__":
    process_csv(
        "./data/Clotho/evaluation_gen/clotho_captions_evaluation.csv",
        Path("./data/Clotho/evaluation_gen"),
    )
