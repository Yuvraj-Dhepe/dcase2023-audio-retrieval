# Language-based Audio Retrieval in DCASE 2024 Challenge

## Quick Start

### 1. Check out source code and install required python packages:

```bash
git clone https://github.com/Yuvraj-Dhepe/dcase2023-audio-retrieval.git
mamba create -f environment.yml
pip install -r requirements.txt # The xformers library should be installed at the end manually, with version mentioned.
cp .env.template .env #Modify the variables as per the system
export $(xargs<.env) #To export the variables before runnign the experiment
```

### 2. Download the [Clotho](https://zenodo.org/records/4783391) dataset:

- The dataset comes with 3 splits: `development`, `validation` and `evaluation`
- Few audio file names and captions were corrected as mentioned in `2.1`
- Curated captions are put in [curated_clotho_captions](./curated_clotho_captions), and are used for all the experiments
  - This folder also contains the csv converted to `row-raw_text-text` format named as {split}\_captions.csv
  - The conversion can be achieved by feeding `clotho_captions_{split}.csv` file and running `cell block 2` in [audio_analysis_nb](./nbs/audio_analysis.ipynb)
  - Make sure to **pass the correct** csv files
  - {split}\_caption.csv file is to be put in Clotho folder, along with extracted audio
  - Directory structure will look as follows after extraction
  - Move all of it to a `data` folder

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

Captions present in `clotho_captions_{split}.csv` are curated captions, which were applied some manual filtering as follows:

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
  - A separate environment needs to be setup for each generative models, change the env_name in  `env.yaml` and install corresponding [audio_gen_requirements.txt](./audio_gen_requirements.txt) or [ez_audio_requirements.txt](./ezaudio_requirements.txt)
  - Run [audio_gen](./audio_generation/audio_gen_new.py) file after modifying the caption column number
    - For audio_gen it takes roughly 48 hrs to generate a single column data, thus generating all columns is a week long process
    - As the weights of model are being pulled from hfspace, one might need to provide their `hf_token`, for the model to be downloaded properly
    - The script will generate audios in batches of 500 for the each caption in the given caption column number in the [clotho_captions_development.csv](./curated_clotho_captions/clotho_captions_development.csv)
    - After all audio's are generated move them to a single folder namely: `Clotho_caption_{col_num}/development`
  - For generating audio with ez_audio model, one has to clone their [repo](https://github.com/haidog-yaqub/EzAudio.git)
    - Install their requirements as mentioned above in a new environment
    - Copy the [ez_audio_gen.py](./audio_generation/ez_audio_gen.py) to the cloned repo, make sure the absolute paths to captions are correct, and then start generating the audios
    - As the weights of model are being pulled from hfspace, one might need to provide their `hf_token`, for the model to be downloaded properly
    - Ezaudio it takes 36 hrs to generate a single column data
    - The script will generate audios in batches of 500 for the each caption in the given caption column number in the [clotho_captions_development.csv](./curated_clotho_captions/clotho_captions_development.csv)
    - After all audio's are generated move them to a single folder namely: `ez_clotho_caption_{col_num}/development`

- For each development split generated audio we need to modify development_captions.csv, so that it includes the name of generated audio and it's corresponding caption text - For this running 3 blocks (1 to 3) in [audio_analysis_nb](./nbs/audio_analysis.ipynb) will be helpful
- This will generate `{split}_captions` for each of the 5 columns into their respective `Clotho_caption_{col_num}` folder
- We will utilize the same files for preprocessing, for both generative models' data, as the names of audios haven't been changed, likewise the captions are also constant. We only change the directory path where generated audios' are stored.


### 3. Preprocessing Audio n Caption Data
- We will utilize the files in [folder](./random_selection_based_preprocessing/) for preprocessing
- [01](./random_selection_based_preprocessing/01_agen_clotho_dataset.py) generates unique caption (tids) and audio ids (fids) based on caption/audio data, and stores them in the respective folder
  - Modify the replication factor, and follow along the comments to preprocess the text and audios
- [02](./random_selection_based_preprocessing/02_agen_multiprocessing_audio_logmel.py) for generating logmels for the audios
  - Follow along the comments and modify the replication factor
- [03](./random_selection_based_preprocessing/03_agen_sbert_embeddings.py) for generating sbert embeddings for the captions
  - Follow along the comments
- After running the preprocessing files, the structure will be as [dir_structure](./dir_struct.md), where the subfolders hold the generated and original data

### 4. Evaluating the synthetic data quality
- To compare the synthetic data quality run the following:
```bash
nohup bash run_audio_comp.sh > ./logs/comp_audios.log 2>&1 &
```
- To generate the plots utilize the [audio_gen audio comparison nb](./nbs/audio_comparison_audio_gen.ipynb) & [ez_audio audio comparison nb](./nbs/audio_comparison_ez_audio.ipynb)

### 5. Training the Baseline
- Get the audio encoder weights from the PANNS zenodo [link](https://zenodo.org/records/3987831), download the Cnn14_mAP=0.431.pth weights, or directly use `wget https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1` and put it in `pretrained_models_weights` directory by renaming it to `cnn14.pth`
- We utilize wandb for tracking the experiment runs, hence one needs to have wanbd account [setup](https://docs.wandb.ai/quickstart/)
- For running the experiments, we need to have a conf-yaml with hyperparams and data directories.
- All the confs used during experiments are present in:
  [conf_yamls](./conf_yamls/) folder.
  - `base_configs` represent the configs used with original clotho data as well as audio_gens generated data
  - `ez_confs` represent the configs used with ezaudios generated data
  - `sweep_configs` represent the configs to perform hyperparameter tuning
  - [hpt_config](./conf_yamls/sweep_configs/best_config.yaml) represents hpt obtained best params

#### 5.1 Training the baseline with original data
- Once the confs is setup one can train a baseline model with original data using the following command:
```bash
python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/base_configs/cap_0_conf.yaml > logs/"${date_time}_Original_baseline.log" 2>&1
```

#### 5.2 Performing HPT on the baseline
- To run the sweeps one have to run the following command:
```bash
python main_wandb_new.py sweep --base_conf_path conf_yamls/sweep_configs/universal_base_config.yaml --sweep_conf_path conf_yamls/sweep_configs/sweep.yaml --run_count 150 --project_name sweeps
```
  - This will create hpt runs under sweeps project inside wandb dashboard
    - It took about 4 to 5 days on `Nvidia RTX 3090 ti` gpu to complete the sweeps

#### 5.3 Using HPT Params with the baseline
- Once the HPT was performed we got the best params as mentioned in [yaml](./conf_yamls/sweep_configs/best_config.yaml), to use these to train the baseline one can run:
```bash
python main_wandb_new.py resume-or-new-run --base_conf_path conf_yamls/sweep_configs/best_config.yaml > logs/"${date_time}_Original_HPT_baseline.log" 2>&1
```

#### 5.4 Using HPT Params and Synthetic Data with baseline
- Once the baseline is established, next experiment including synthetic data can be performed
- For capturing the variation in baseline model with synthetic data, incremental synthetic data subsets from 1x to 5x along with original data are chosen to train the model
- The data should be ready, from the steps mentioned in sec 3
- To run the model trainings simply call the bash script of run.sh
```bash
nohup bash run.sh > ./logs/synth_runs.log 2>&1 &
```
- Roughly it takes about 3 days to complete all of the model runs
- All the model evaluation is automated in the python script, which evaluates the model on the evaluation set, after the training of the model is completed.

With all the above procedures followed, we complete the major experiments of the thesis. Lastly to check the influence of random_seeds, one has to follow the following steps:

#### 6 Random Seed Experiments
