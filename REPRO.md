# 🔁 Reproducibility Instructions

This document provides the full set of instructions to reproduce our project results from scratch, including data setup, environment configuration, training, and evaluation.

---

## 🧱 Project Structure

```bash
.
├── seater/                    # All source code (models, training, evaluation)
├── requirements.txt        # Python dependencies
├── README.md               # README file
├── REPRO.md                # This file
├── jobs/                   # Slurm files for snellius
```

---

## ⚙️ Environment Setup


Setup project by running ```jobs\install_environment.job```. For a local installation you can run the commands in that file one by one.
To check your installation and set up wandb open ```jobs\check_environment.job``` and put in your wandb api key, then run it.



---

## 📂 Download & Prepare Datasets

Download the books data linked in the original readme, then make sure you update the ''datafile_par_path'' in ```config\const.py``` to point to that directory.

For processing new datasets we have prepared some notebooks in ```\data\```. Note that these area a starting point, and are not fully functional out of the box.

The raw data for the books dataset can be downloaded from https://nijianmo.github.io/amazon/. The paper uses the 5-core version. The book information (for fairness metrics) can be gotten from the metadata file. See ```data\Books\preprocess_categories.ipynb``` for how to use these files.

Note: The link to the books data is down as I am writing this, but the file can also be downloaded from this page: https://amazon-reviews-2023.github.io/




---

## ⚙️ Configuration

Set your parameters in the config file before training. Example:


I don't know what the maximum parameters are to make it work with snellius, but the default values were too big. Here is what I changed:
Reduced num_workers 32->16.
Reduced batch_size 2048->1024. 
Reduced test_batch_size 2048->256. 

Most of the parameters are added as flags in the run file.



---

## 🚀 5. Training



Run the following command to train the baseline:

```jobs\run.job```

### Baselines

You can change the model flag to run both SASREC and SEATER. When running SEATER on a new dataset you will first have to train a SASREC model and extract embeddings from it.



## 📈 Evaluation

After training you can run evaluation only by commenting out the training step in ```main.py```.

The model is evaluated using ```utils\metrics.py```. To the utils I have added code for the beyond accuracy metrics IDL and GINI. Given the required data (item representation/categories) you can easily add these to the evaluation in the metrics file.

---


## 📎 Misc. Notes (optional)

---

## 📦 Dependencies / References

This project repository uses the following frameworks / refers to the following papers:

- SEATER: https://github.com/Ethan00Si/SEATER_Generative_Retrieval


