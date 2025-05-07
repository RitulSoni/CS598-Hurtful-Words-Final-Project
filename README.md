# Reproducing and Extending Bias Analysis in Clinical Contextual Word Embeddings

This repository contains the code and report for a final project for CS 598 [CS598 Deep Learning for Healthcare] at the University of Illinois Urbana-Champaign, completed by Ritul Soni (rsoni27@illinois.edu).

## Overview

This project reproduces core methodologies from the paper "Hurtful Words: Quantifying Biases in Clinical Contextual Word Embeddings" by Zhang et al. (2020). We focus on quantifying gender bias within the `Baseline_Clinical_BERT` model provided by the original authors, using the full MIMIC-III clinical dataset (v1.4).

The project involves:
1.  **Data Preprocessing:** Loading and cleaning MIMIC-III CSVs, including processing `NOTEEVENTS.csv` to incorporate clinical text, calculating age, cleaning demographics, and aggregating ICD codes.
2.  **Bias Quantification:** Implementing Algorithm 1 from the paper to calculate prior-adjusted log probability bias scores using custom templates for "diabetes" and "pneumonia".
3.  **Downstream Task:** Setting up an in-hospital mortality prediction task using a sample of processed notes, extracting BERT embeddings, training a Logistic Regression classifier.
4.  **Fairness Evaluation:** Calculating standard fairness metrics (Demographic Parity, Recall Gap, Specificity Gap, FPR Gap) and an extension (Precision Gap) based on gender for the downstream task.
5.  **Analysis & Visualization:** Comparing results to the original paper (where applicable) and discussing findings and reproducibility.

## Original Paper

Zhang, H., Lu, A. X., Abdalla, M., McDermott, M. B., & Ghassemi, M. (2020). Hurtful words: Quantifying biases in clinical contextual word embeddings. *Proceedings of the ACM Conference on Health, Inference, and Learning (CHIL)*, 110–120.
* **Paper Link:** [https://doi.org/10.1145/3368555.3384448](https://doi.org/10.1145/3368555.3384448)
* **Original Code/Model Repository:** [https://github.com/MLforHealth/HurtfulWords](https://github.com/MLforHealth/HurtfulWords)

## Repository Contents

* `data_preprocesssing.ipynb`: Jupyter notebook for loading and preprocessing MIMIC-III CSV data.
* `CS598_Final_Project.ipynb`: Jupyter notebook for loading the pre-trained model, calculating template bias scores, running the downstream task, evaluating fairness, and generating results/plots.
* `report.pdf`: (Recommended) Include the final compiled PDF report here.
* `references.bib`: Bibliography file for the report.
* `requirements.txt`: Python package dependencies.
* `README.md`: This file.

## Requirements

### 1. Data Access
* **MIMIC-III Clinical Database v1.4:** Access requires completion of the CITI "Data or Specimens Only Research" course and signing a Data Use Agreement via PhysioNet.
    * Download instructions: [https://physionet.org/content/mimiciii/1.4/](https://physionet.org/content/mimiciii/1.4/)
    * You will need the full CSV files (specifically `PATIENTS.csv`, `ADMISSIONS.csv`, `DIAGNOSES_ICD.csv`, and `NOTEEVENTS.csv`).

### 2. Model Access
* **Baseline\_Clinical\_BERT Model:** Download the `baseline_clinical_BERT_1_epoch_512.tar` file from the original authors' repository: [https://github.com/MLforHealth/HurtfulWords](https://github.com/MLforHealth/HurtfulWords) (check their instructions/links for model weights).

### 3. Software
* **Python:** Version 3.11.12 (as used in Colab)
* **Key Libraries:** See `requirements.txt`. Install using:
    ```bash
    pip install -r requirements.txt
    ```
* **Environment:** Developed and tested in Google Colab.

### 4. Hardware
* **GPU:** Required for efficient BERT embedding extraction. Google Colab with a T4 GPU was used.
* **RAM:** Significant RAM may be needed for loading and processing MIMIC-III CSVs, especially `NOTEEVENTS.csv`. The preprocessing notebook uses techniques suitable for Colab's RAM limits but might require adjustments based on the exact Colab instance or local machine specs.

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/RitulSoni/CS598-Hurtful-Words-Final-Project.git](https://github.com/RitulSoni/CS598-Hurtful-Words-Final-Project.git)
    cd CS598-Hurtful-Words-Final-Project
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Organize Data and Model:**
    * Create a directory structure within the repository or on your connected Google Drive. The notebooks assume a structure like this (you **must** adapt paths within the notebooks if your structure differs):
        ```
        CS598-Hurtful-Words-Final-Project/
        ├── CS598_Final_Project.ipynb
        ├── data_preprocesssing.ipynb
        ├── report.pdf
        ├── references.bib
        ├── requirements.txt
        ├── README.md
        ├── data/                 # Create this directory
        │   └── mimic-iii-clinical-database-1.4/  # Place MIMIC-III CSVs here
        │       ├── PATIENTS.csv
        │       ├── ADMISSIONS.csv
        │       ├── DIAGNOSES_ICD.csv
        │       └── NOTEEVENTS.csv
        ├── model/                # Create this directory
        │   └── baseline_clinical_BERT_1_epoch_512.tar # Place downloaded model here
        └── output/               # Notebooks will save outputs here (plots, processed data)
            ├── processed_mimic_full_subset.csv
            └── processed_mimic_full_subset.pkl
            └── template_bias_scores_plot.png
            └── ... (other plots)
        ```
    * **Important:** The notebooks use paths relative to Google Drive (e.g., `/content/drive/MyDrive/CS598 Final Project/...`). You **must** either:
        * Replicate this structure in your Google Drive and mount it in Colab.
        * Modify the file paths within both notebooks to match your local or Colab storage structure. Pay attention to `BASE_DATA_PATH`, `TAR_FILE_PATH`, `EXTRACTION_DIR_ON_DRIVE`, `OUTPUT_PROCESSED_DIR`, etc.

## Running the Code

**Note:** Running the full preprocessing and analysis requires significant time (especially data loading/processing and embedding extraction) and computational resources (RAM, GPU).

1.  **Step 1: Data Preprocessing**
    * Open and run the `data_preprocesssing.ipynb` notebook.
    * Ensure the paths to your MIMIC-III CSVs are correctly set in Block 2.
    * **Potential Issue:** Loading `NOTEEVENTS.csv` (Block 2) might fail with a `ParserError` or memory issues due to its size, even with chunking/sampling. Try adjusting `chunksize` or ensure the file isn't corrupted. If using the full dataset, ensure sufficient RAM. The `ParserError` you encountered previously needs resolution - it might require `engine='python'` in `pd.read_csv` for that specific file if parsing fails, though this is slower.
    * This notebook performs merging, cleaning, age calculation, etc., and saves the final processed data to `output/processed_mimic_full_subset.csv` and `output/processed_mimic_full_subset.pkl` (or the paths you specified). This step needs to complete successfully before proceeding.

2.  **Step 2: Bias Analysis and Downstream Task**
    * Open and run the `CS598_Final_Project.ipynb` notebook.
    * Ensure Google Drive is mounted (Block 2) if using Drive for storage.
    * Ensure the path to the downloaded `baseline_clinical_BERT_1_epoch_512.tar` file is correct (Block 5, `TAR_FILE_PATH`). The notebook extracts this model.
    * Ensure the path to the processed data (`processed_mimic_full_subset.csv`) generated in Step 1 is correct (Block 9, `pd.read_csv(...)`).
    * Running the embedding extraction (Block 10) will require a GPU and takes time (~15-20 mins for the 3k sample on a T4).
    * The notebook calculates template bias scores, trains the downstream classifier, evaluates performance and fairness, and generates plots (which should be saved to disk if you add `plt.savefig('output/figure_name.png')` calls after `plt.show()` in Blocks 15, 16, 17).

## Results

The primary results (bias scores, performance metrics, fairness gaps) are printed within the `CS598_Final_Project.ipynb` notebook outputs and summarized in the final report (`report.pdf`). Visualizations are generated by the notebook and should be saved as `.png` files in the `output/` directory (or similar).

## Citation

If using this code or findings, please cite the original "Hurtful Words" paper:

```bibtex
@inproceedings{zhang2020hurtful,
 title={Hurtful words: Quantifying biases in clinical contextual word embeddings},
 author={Zhang, Haoran and Lu, Amy X and Abdalla, Mohamed and McDermott, Matthew and Ghassemi, Marzyeh},
 booktitle={Proceedings of the ACM Conference on Health, Inference, and Learning},
 pages={110--120},
 year={2020}
}
