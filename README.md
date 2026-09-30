# Fake News Detection System

A machine-learning prototype that classifies English-language news article text as **fake** or **authentic**. It combines TF-IDF text features with four scikit-learn classifiers and a Streamlit interface for interactive predictions and model explanations.

> [!IMPORTANT]
> This is an educational decision-support prototype, not a fact-checking authority. A model prediction is a learned language-pattern association, not evidence that an article or claim is true or false. Always verify claims with reliable sources and human review.

![Feature importance chart](images/feature_importance.png)

## Features

- Classifies pasted article text with Logistic Regression, calibrated Linear SVM, Random Forest, and Multinomial Naive Bayes.
- Builds TF-IDF features from unigrams and bigrams.
- Displays prediction scores and word-level explanations in the web app.
- Includes training, evaluation, data inspection, and chart-generation scripts.

## Project Structure

```text
.
├── app.py                         # Streamlit application
├── train_models.py                # Train models and generate evaluation artifacts
├── inspect_data.py                # Inspect input datasets
├── generate_report_charts.py      # Generate report charts
├── generate_benchmark_metrics.py  # Generate benchmark metrics
├── requirements.txt
├── data/
│   ├── Fake.csv                   # Labeled fake-news examples
│   └── True.csv                   # Labeled authentic-news examples
├── models/                        # Saved model and vectorizer files
├── src/                           # Preprocessing, model, and explanation modules
├── images/                        # Project figures
└── notebooks/                     # Project notebooks
```

## Requirements

- Python 3.10 or newer
- pip

## Setup and Run

From the repository root, create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Train models and create the metrics and feature-importance files used by the app:

```bash
python train_models.py
```

The repository currently includes saved model files, but retraining regenerates model artifacts and evaluation outputs from the CSV datasets.

Launch the application:

```bash
python -m streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`, in the terminal.

## Data and Model Details

The training script reads `data/Fake.csv` and `data/True.csv`, assigns fake examples label `1` and authentic examples label `0`, removes rows with missing or empty article text, and uses a stratified 80/20 train-test split. TF-IDF is configured for up to 5,000 features, unigrams and bigrams, English stop words, and document-frequency limits. Four classifiers are trained and saved under `models/`.

The datasets are commonly distributed as the ISOT Fake News Dataset. The project report cites the [Kaggle dataset repository](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset). Check the source dataset's terms and attribution requirements before redistributing the CSV files.

## Evaluation Status

**Do not treat the currently generated benchmark scores as valid model performance.** In `train_models.py`, the evaluation function deliberately flips selected test predictions before calculating metrics. The TF-IDF vectorizer is also fitted on the full dataset before the train-test split, allowing test-set vocabulary information into training. These issues should be corrected and metrics regenerated before reporting accuracy, precision, recall, F1, ROC-AUC, confusion matrices, or comparisons.

## Responsible Use and Limitations

- The classifier learns patterns in its labeled training data; it does not verify claims against primary sources.
- Predictions can reflect dataset biases, topics, sources, time period, and labeling quality.
- Writing style, satire, political perspective, and unfamiliar topics can lead to errors.
- Use outputs only to prioritize material for review, never as a basis for automatic moderation or factual judgments without human verification.

## License

No license file is currently provided. Add a license before permitting reuse or redistribution of this project.
