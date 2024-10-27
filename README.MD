# Language-based Audio Retrieval in DCASE 2024 Challenge

## Quick Start

### 1. Check out source code and install required python packages:

```
git clone https://github.com/Yuvraj-Dhepe/dcase2023-audio-retrieval.git
mamba create -f environment.yml
pip install -r requirements.txt
```

### 2. Download the [Clotho](https://zenodo.org/records/4783391) dataset:

- The dataset comes with 3 splits: `development`, `validation` and `evaluation`
- Few audio file names and captions were corrected as mentioned in `2.1`
- Curated captions are put in [curated_clotho_captions](./curated_clotho_captions)
  - This folder also contains the csv converted to `row-raw_text-text` format named as {split}\_captions.csv
  - The conversion can be achieved by feeding `clotho_captions_{split}.csv` file and running `cell block 2` in [audio_analysis_nb](./audio_analysis.ipynb)
  - {split}\_caption.csv file is to be put in Clotho folder, along with extracted audio
  - Directory structure will look as follows

```
Clotho
├─ development_captions.csv
├─ validation_captions.csv
├─ evaluation_captions.csv
├─ development
│   └─...(3839 wavs)
├─ validation
│   └─...(1045 wavs)
└─ evaluation
    └─...(1045 wavs)
```

#### 2.1 Caption Corrections:

Captions present in `clotho_captions_{split}.csv` were applied some manual filtering as follows:

- echos replaced with echoes
- removal of dual-spaces, present in few captions
- NOTE: Some captions have words spelled in British English form: for example
  > Gravelled (British) <-> Graveled (American)
  > Neighbourhood (British) <-> Neighborhood (American)
  > Signalling (British) <-> Signaling (American)
  > We keep them as is.
- removal of double words like `a a`, `the the` `busy busy`, `to to`, `on on`, 'bicycle bicycle' where there was no context for the words presence

#### 2.2 Audio Corrections

- When the audio files from archive's are extracted you can see the very first files do have a `space character` in their name, these spaces were removed from the audio file name, as well as it's corresponding row in file*name column in `clotho_captions*{split}.csv`

#### 2.3 Audio Generations & Caption csv modifications

- Following steps need to be followed for audio generation

  - Run [audio_gen](./audio_gen_new.py) file after modifying the caption column number\
  - The script will generate audios in batches of 500 for the each caption in the given caption column number in the `clotho_captions_{split}.csv`
  - A single audio generation usually takes 48s, for a single column it takes around 48 hrs
  - P.S. Currently only development split captions are used in data generation

- For each development split generated audio we need to modify development_captions.csv, so that it includes the name of generated audio and it's corresponding
  caption text - For this running all first 3 blocks in [audio_analysis_nb](./audio_analysis.ipynb) will be helpful

### 3. Preprocessing Audio n Caption Data

- Run the files in the following order, follow the comments in the respctive files
- `agen_` prefix denotes the files used for processing audio and captions after generating the synthetic data
- `old_` prefix files can be used if one is working with original data rather than generated data
- Get the audio encoder weights from the PANNS zenodo [link](https://zenodo.org/records/3987831), download the Cnn14_mAP=0.431.pth weights, or directly use `wget https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1` and put it in pretrained models by naming it as cnn14.pth
- NOTE:
  - Run the files in the numerical order given (only one of files with same numebr prefix should be used in one run)
  - Change the dataset_dirs properly in the files before running them to ensure, you are processing correct audio and captions

| File Name                                 | Description                                                                                                                                                      |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `01_agen_clotho_dataset.py`               | Process audio captions, generate fids and tids                                                                                                                   |
| `02_agen_multiprocessing_audio_logmel.py` | Extract log-mel energies from audio clips using multiprocessing                                                                                                  |
| `02_gpu_audio_logmel.py`                  | GPU-based version for logmel extraction (running on GPU, can change logmel representation slightly due to GPU number representation differences compared to CPU) |
| `02_multiprocessing_audio_logmel.py`      | Extract log-mel energies from audio clips using multiprocessing                                                                                                  |
| `03_agen_sbert_embeddings.py`             | Generate sentence embeddings using Sentence-BERT (all-mpnet-base-v2)                                                                                             |
| `04_cnn14_transfer.py`                    | Transfer pretrained CNN14 (Cnn14_mAP=0.431.pth), for this                                                                                                        |
| `old_audio_logmel.py`                     | (Old version) Script for audio logmel extraction                                                                                                                 |
| `old_clotho_dataset.py`                   | (Old version) Script for processing Clotho dataset                                                                                                               |
| `old_sbert_embeddings.py`                 | (Old version) Script for generating sentence embeddings using SBERT                                                                                              |
