# Semi-LAR: Semi-Supervised Nighttime Lens Flare Removal

Lens flare is a common degradation in nighttime photography, especially when a camera faces strong light sources. Flare artifacts such as glare, streaks, and ghosting can reduce image contrast, hide scene details, and make downstream
vision tasks less reliable. This repository provides a semi-supervised flare removal framework built around a teacher-student pipeline.

## Table of Contents

- [Semi-LAR: Semi-Supervised Nighttime Lens Flare Removal](#semi-lar-semi-supervised-nighttime-lens-flare-removal)
  - [Installation](#installation)
  - [Dataset](#dataset)
  - [Training](#training)
    - [Initialize Candidate Repository](#initialize-candidate-repository)
    - [Training from Scratch](#training-from-scratch)
    - [Resume Training](#resume-training)
  - [Testing](#testing)
    - [Saved Checkpoint](#saved-checkpoint)
    - [Inference](#inference)
    - [Evaluation](#evaluation)
  - [Project Structure](#project-structure)
  - [Notes](#notes)
  - [License](#license)
  - [Acknowledgement](#acknowledgement)

## Installation

1. Clone the repository.

```bash
git clone https://github.com/copr-sans/Semi-LAR.git
cd Semi-LAR
```

2. Create a Python environment.

```bash
conda create -n semi-lar python=3.10
conda activate semi-lar
```

3. Install the remaining requirements.

```bash
pip install -r requirements.txt
```

## Dataset

By default, the training script reads data from `./data`. Please organize the dataset as follows:

```text
data
├── labeled
│   ├── input
│   ├── LA
│   └── GT
├── unlabeled
│   ├── input
│   └── candidate
├── val
│   ├── input
│   └── GT
└── test
    └── real
        ├── input
        ├── gt
        └── mask
```

Dataset download: [Baidu Netdisk](https://pan.baidu.com/s/1y9fZKDnoFEuMLDVwVC19rg?pwd=gv62), extraction code: `gv62`.

If you want to generate the corresponding labeled dataset yourself, you can refer to the `Generate_flare_on_light` folder in the [ykdai/Flare7K](https://github.com/ykdai/Flare7K) project.

## Training

### Initialize Candidate Repository

Before semi-supervised training, initialize the candidate repository for
unlabeled images:

```bash
python create_candidate.py
```

Make sure the `candidate` folder exists before running the script.

### Training from Scratch

To train the model from scratch, run:

```bash
python train.py
```

You can also configure common training parameters from the command line:

```bash
python train.py \
  --data_dir ./data \
  --num_epochs 40 \
  --train_batchsize 4 \
  --val_batchsize 2 \
  --crop_size 512 \
  --num_workers 4 \
  --save_path ./experiments/model_RaLiFormer/ckpt/ \
  --log_dir ./experiments/model_RaLiFormer/log \
```

### Resume Training

To resume from a specific checkpoint, run:

```bash
python train.py \
  --resume True \
  --resume_path ./experiments/model_RaLiFormer/ckpt/model_e10.pth
```

## Testing

### Saved Checkpoint

The default inference script loads:

```text
./experiments/model_RaLiFormer/ckpt/model_best.pth
```

Checkpoint download: [Baidu Netdisk](https://pan.baidu.com/s/1AX2daPolgsm6tWEHqrlOow), extraction code: `cd6v`.

Place your trained checkpoint at this path, or update `model_root` in `test.py`
before running inference.

### Inference

To run inference on the real test set:

```bash
python test.py
```

### Evaluation

To evaluate restored images with PSNR, SSIM, LPIPS, glare PSNR, and streak PSNR,
run:

```bash
python evaluate.py \
  --input ./result/real/images \
  --gt ./data/test/real/gt \
  --mask ./data/test/real/mask
```

The script reports average image quality metrics in the terminal.

## License

This project is licensed under the S-Lab License 1.0. Redistribution and use
should follow this license.

## Acknowledgement

This project is related to nighttime lens flare removal research and uses common
evaluation settings from flare removal benchmarks such as Flare7K++. We thank
the open-source image restoration and flare removal communities for their
datasets, codebases, and research contributions.
