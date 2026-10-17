

## Installation

The code requires Python, PyTorch, and a CUDA-capable GPU. Install the remaining dependencies with:

```bash
pip install -r requirements.txt
```

For a CUDA-specific PyTorch build, follow the installation command provided for your CUDA version before installing the remaining packages.

## Dataset Preparation

Place the three datasets under one root directory. Each text file contains image identifiers without filename extensions.

```text
datasets/
├── IRSTD-1k/
│   ├── images/
│   ├── masks/
│   ├── trainval.txt
│   └── test.txt
├── NUDT-SIRST/
│   ├── images/
│   ├── masks/
│   ├── trainval.txt
│   └── test.txt
└── NUAA-SIRST/
    ├── images/
    ├── masks/
    ├── trainval.txt
    └── test.txt
```

Images and masks are expected to have matching `.png` filenames. All samples are processed at 512 x 512 resolution.

## Training

Use `--dataset-name` to select a dataset and `--data-root` to specify the common dataset root. For example:

```bash
python main.py \
  --mode train \
  --dataset-name IRSTD-1k \
  --data-root ./datasets \
  --save-root ./runs \
  --batch-size 4 \
  --epochs 1000 \
  --lr 1e-3
```


## Evaluation

Download the corresponding checkpoint from the table above and run:

```bash
python main.py \
  --mode test \
  --dataset-name IRSTD-1k \
  --data-root ./datasets \
  --weight-path 
```



## Repository Structure

```text
.
├── assets/                 # Figures used by this README
├── model/
│   ├── modules/
│   │   ├── msad.py         # Multi-Scale Anti-Alias Downsampling
│   │   └── pgsm.py         # Prototype-Guided Skip Memory
│   └── rdanet.py           # RDANet architecture
├── utils/
│   ├── data.py             # Dataset loading and preprocessing
│   ├── losses.py           # Training losses
│   └── metric.py           # Evaluation metrics
├── main.py                 # Training and evaluation entry point
└── requirements.txt
```

