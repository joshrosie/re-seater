# 🔁 Reproducibility Instructions

This document provides the full set of instructions to reproduce our project results from scratch, including data setup, environment configuration, training, and evaluation.

---

## 🧱 Project Structure

```bash
.
├── data/                   # Contains raw and processed datasets
├── src/                    # All source code (models, training, evaluation)
├── requirements.txt        # Python dependencies
├── README.md               # README file
├── REPRO.md                # This file
├── XXXXX
├── XXXXX
```

---

## ⚙️ Environment Setup


Setup project by running the following commands:



```#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=InstallEnvironment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=/home/_snellius-name_/logs/out-%x.%A.out
#SBATCH --error=/home/_snellius-name_/logs/err-%x.%A.err

module purge
module load 2024
module load Miniconda3/24.7.1-0
module load jax/0.3.14-foss-2022a-CUDA-11.7.0
module load CUDA/12.6.0

# Go to the directory that contains the conda env file and install it
cd $HOME/seater
conda create -n seater python=3.9.16 -y
source activate seater
pip install -r requirements.txt
conda install -c conda-forge cudatoolkit=11.1
```

---

## 📂 Download & Prepare Datasets

Place your datasets in the `XXXX/` directory.

### Example Dataset
```bash
mkdir -p data/example_dataset
cd data/example_dataset
wget xxxxx
python -m src.preprocess_example_dataset.py xxxx
cd ../..
```

---

## ⚙️ Configuration

Set your parameters in the config file before training. Example:


---

## 🚀 5. Training

### Baselines

Run the following command to train the baseline:

```bash
python XXXX
```

To perform inference:

```bash
python XXXX
```

Alternatively, execute the following slurm jobs:

```bash
sbatch job_scripts/train_xxxxx.job
sbatch job_scripts/infer_xxxxx.job
```

---

## 📈 Evaluation

After training, evaluate all models with:

```bash
python XXXX
```

---


## 📎 Misc. Notes (optional)

---

## 📦 Dependencies / References

This project repository uses the following frameworks / refers to the following papers:

- XXX
- XXX


