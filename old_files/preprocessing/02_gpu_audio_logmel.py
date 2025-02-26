import glob
import os
import pickle
import h5py
import librosa
import numpy as np
import torch
from torch import nn


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate=44100,
        window_length_secs=0.025,
        hop_length_secs=0.010,
        num_mels=128,
        log_offset=0.0,
    ):
        super(LogMelSpectrogram, self).__init__()
        self.sample_rate = sample_rate
        self.window_length = int(round(sample_rate * window_length_secs))
        self.hop_length = int(round(sample_rate * hop_length_secs))
        self.fft_length = 2 ** int(
            np.ceil(np.log(self.window_length) / np.log(2.0))
        )
        self.num_mels = num_mels
        self.log_offset = log_offset

        self.mel_filter_bank = nn.Parameter(
            torch.tensor(
                librosa.filters.mel(
                    sr=sample_rate, n_fft=self.fft_length, n_mels=num_mels
                ),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

    def forward(self, y):
        y = y.unsqueeze(0)  # Add batch dimension
        spectrogram = torch.stft(
            y,
            n_fft=self.fft_length,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=torch.hann_window(self.window_length).to(y.device),
            return_complex=True,
        )
        spectrogram = torch.abs(spectrogram) ** 2
        mel_spectrogram = torch.matmul(self.mel_filter_bank, spectrogram)
        log_mel_spectrogram = torch.log(mel_spectrogram + self.log_offset)
        return log_mel_spectrogram


def process_audio(fpath, fname2fid, device):
    fname = os.path.basename(fpath)
    fid = fname2fid[fname]

    # print(f"Processing {fname}...")

    y, sr = librosa.load(fpath, sr=None, mono=True)

    # Add assert statement to check if audio is mono
    assert len(y.shape) == 1, "Audio is not mono!"

    y = torch.tensor(y, dtype=torch.float32).to(device)
    assert sr == 44100, print("NOT 44100 SR")
    log_mel_model = LogMelSpectrogram(
        sample_rate=sr,
        window_length_secs=0.040,
        hop_length_secs=0.020,
        num_mels=64,
        log_offset=np.spacing(1),
    )
    log_mel_model.to(device)

    with torch.no_grad():
        log_mel = log_mel_model(y).squeeze().cpu().numpy()

    # print(f"Finished processing {fname}...")

    return fid, log_mel.transpose()


global_params = {
    "dataset_dir": "./data/Clotho",
    "audio_splits": ["development", "validation", "evaluation"],
}

# Load audio info
audio_info = os.path.join(global_params["dataset_dir"], "audio_info.pkl")
with open(audio_info, "rb") as store:
    audio_fid2fname = pickle.load(store)["audio_fid2fname"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Extract log mel for splits
for split in global_params["audio_splits"]:
    fid2fname = audio_fid2fname[split]
    fname2fid = {fid2fname[fid]: fid for fid in fid2fname}

    audio_dir = os.path.join(global_params["dataset_dir"], split)
    audio_logmel = os.path.join(
        global_params["dataset_dir"], f"{split}_audio_logmels.hdf5"
    )

    with h5py.File(audio_logmel, "w") as stream:
        audio_files = glob.glob(r"{}/*.wav".format(audio_dir))
        for audio_file in audio_files:
            fid, log_mel = process_audio(audio_file, fname2fid, device)
            stream[fid] = log_mel

    print(f"Saved {audio_logmel}")
