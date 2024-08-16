# Language-based Audio Retrieval in DCASE 2023 Challenge

This repository provides the baseline system for **Language-based Audio Retrieval** (Task 6B) in DCASE 2023 Challenge.

**2023/03/20 Update:**
Training checkpoints for the baseline system and its audio encoder are available on Zenodo:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7752975.svg)](https://doi.org/10.5281/zenodo.7752975).

![Language-based Audio Retrieval](figs/dcase2023_task_6b.png)

# Baseline Retrieval System

![Baseline Retrieval System](figs/baseline_system.png)

```
- Audio Encoder                   # fine-tuned PANNs, i.e., CNN14
- Text Encoder                    # pretrained Sentence-BERT, i.e., all-mpnet-base-v2
- Contrastive Learning Objective  # InfoNCE loss
```

# Quick Start

This codebase is developed with Python 3.9 and [PyTorch 1.13.0](https://pytorch.org/).

1. Check out source code and install required python packages:

```
git clone https://github.com/xieh97/dcase2023-audio-retrieval.git
pip install -r requirements.txt
```

2. Download the [Clotho](https://zenodo.org/record/4783391) dataset:

```
Clotho
├─ clotho_captions_development.csv
├─ clotho_captions_validation.csv
├─ clotho_captions_evaluation.csv
├─ development
│   └─...(3839 wavs)
├─ validation
│   └─...(1045 wavs)
└─ evaluation
    └─...(1045 wavs)
```

3. Pre-process audio and caption data: `Run the files in the given sr. order`

```
preprocessing
├─ 01_clotho_dataset.py            # process audio captions, generate fids and cids
├─ 02_audio_logmel.py              # extract log-mel energies from audio clips, can run audio_logmel_gpu.py for gpu based version
├─ 03_sbert_embeddings.py          # generate sentence embeddings using Sentence-BERT (all-mpnet-base-v2)
└─ 04_cnn14_transfer.py            # transfer pretrained CNN14 (Cnn14_mAP=0.431.pth)
```

4. Train the baseline system:
- Run main_wandb.py for tracking results in wanbdmain.py or
- Run new_main.py an updated version of main.py, however, it throws some ray error, which needs to be corrected
- Get the audio encoder weights from the PANNS zenodo [link](https://zenodo.org/records/3987831), download the Cnn14_mAP=0.431.pth weights, or directly use `wget https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1` and put it in pretrained models by naming it as cnn14.pth


```
models
├─ core.py                      # dual-encoder framework
├─ audio_encoders.py            # audio encoders
└─ text_encoders.py             # text encoders

utils
├─ criterion_utils.py           # loss functions
├─ data_utils.py                # Pytorch dataset classes
└─ model_utils.py               # model.train(), model.eval(), etc.

conf.yaml                       # experimental settings, **Edit the data and model paths before starting the training**
main.py                         # main()
main_wandb.py                   # Running main where results are tracked in wandb
```

5. Calculate retrieval metrics:
`Run the files in the given sr. no order, can run wanbd alternatives to track metrics in wandb`
```
postprocessing
├─ 01_xmodal_scores.py             # calculate audio-text scores
└─ 02_xmodal_retrieval.py          # calculate mAP, R@1, R@5, R@10, etc.
```

# Examples

1. Code example for using the pretrained audio encoder:

```
example
├─ audio_encoder.py             # code example for audio encoder
├─ example.wav                  # audio segment example
└─ audio_encoder.pth            # audio encoder checkpoint (https://doi.org/10.5281/zenodo.7752975)
```