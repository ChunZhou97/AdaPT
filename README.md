## 1. Environment setup

It is recommended to create a clean Python environment first.

You can create a Conda environment and then install the requirements.

```bash
conda create -n adapt python=3.7 -y
conda activate adapt
pip install -r requirements.txt
```

---

## 2. Prepare datasets

Before running the code, make sure your datasets are available locally.

The current scripts use **hard-coded dataset paths** in both `extract_pos.py` and `main.py`:

```python
DATASET_ROOTS = {
    "CIFAR10": "/data/zc/datasets/CIFAR10",
    "CIFAR100": "/data/zc/datasets/CIFAR100",
    "MiniImageNet": "/data/zc/datasets/mini-imagenet",
    "Caltech101": "/data/zc/datasets/caltech-101/101_ObjectCategories",
}
```

Update these paths to match your local machine before running experiments.

### Supported datasets

- **CIFAR-10**
- **CIFAR-100**
- **Mini-ImageNet**
- **Caltech-101**

All images are resized to **224 × 224** inside the dataset pipeline before training/testing.

---

## 3. Extract trigger insertion positions

Run `extract_pos.py` first. This script computes AdaPT trigger positions and saves them into the `positions/` directory.

### Arguments

- `--surrogate_models`: surrogate models used to compute Grad-NAM
  - choices: `resnet101`, `resnet50`, `regnet_x_3_2gf`
- `--dataset_name`: dataset name
  - choices: `CIFAR10`, `CIFAR100`, `MiniImageNet`, `Caltech101`
- `--mode`: which split to process
  - choices: `train`, `test`
- `--trigger_size`: Trigger-Conv kernel size
- `--num_workers`: DataLoader workers
- `--save_dir`: output directory for position files
- `--device`: `cuda` or `cpu`

### Example commands

Extract training positions:

```bash
python extract_pos.py \
  --surrogate_models resnet101 \
  --dataset_name CIFAR10 \
  --mode train \
  --trigger_size 13 \
  --num_workers 4 \
  --save_dir positions \
  --device cuda
```

Extract test positions:

```bash
python extract_pos.py \
  --surrogate_models resnet101 \
  --dataset_name CIFAR10 \
  --mode test \
  --trigger_size 13 \
  --num_workers 4 \
  --save_dir positions \
  --device cuda
```

### Output structure

After running the script, the generated files will look like this:

```text
positions/
└── CIFAR10/
    └── resnet101/
        ├── positions_train.txt
        └── positions_test.txt
```

You must generate the corresponding position files before running `main.py`, otherwise the dataset loader will raise `FileNotFoundError`.

---

## 4. Run poisoned training and backdoor testing

After extracting positions, run `main.py` to train the victim model and evaluate benign accuracy and attack success rate.

### Arguments

- `--learning`: learning mode
  - `trans`: transfer learning
  - `e2e`: end-to-end learning
- `--data`: dataset name
  - `CIFAR10`, `CIFAR100`, `MiniImageNet`, `Caltech101`
- `--loc`: attack type
  - `BadNets`, `Blend`, `AdaPT_BadNets`, `AdaPT_Blend`
- `--surrogate_models`: surrogate model tag used to locate the saved position files
  - `resnet50`, `resnet101`, `regnet_x_3_2gf`
- `--epochs`: number of training epochs
- `--p`: poisoning rate

### Example commands

Transfer learning with AdaPT-BadNets on CIFAR-10:

```bash
python main.py \
  --learning trans \
  --data CIFAR10 \
  --loc AdaPT_BadNets \
  --surrogate_models resnet101 \
  --epochs 100 \
  --p 0.1
```

End-to-end learning with AdaPT-Blend on CIFAR-100:

```bash
python main.py \
  --learning e2e \
  --data CIFAR100 \
  --loc AdaPT_Blend \
  --epochs 100 \
  --p 0.1
```

---

## 5. Minimal running example


### Step A: extract training positions

```bash
python extract_pos.py --surrogate_models resnet101 --dataset_name CIFAR10 --mode train --device cuda
```

### Step B: extract test positions

```bash
python extract_pos.py --surrogate_models resnet101 --dataset_name CIFAR10 --mode test --device cuda
```

### Step C: train and evaluate

```bash
python main.py --learning trans --data CIFAR10 --loc AdaPT_BadNets --surrogate_models resnet101 --epochs 100 --p 0.1
```

---
