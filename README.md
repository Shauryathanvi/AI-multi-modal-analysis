# AI Multi-Modal Analysis

This repository contains end-to-end notebooks for experimenting with machine learning workflows across four data modalities: images, audio, tabular data, and text. Each notebook walks through data preparation, exploratory analysis, classical and deep learning models, and model interpretation.

## Repository Structure
- **Image.ipynb** – Disease image classification using TensorFlow CNNs (MobileNetV2/ResNet50) alongside classical baselines (SVM, Random Forest, KNN), with Grad-CAM visualizations for model explainability.
- **Sound.ipynb** – UrbanSound8K audio classification with a CNN pipeline, classical models, PCA-based clustering, and SHAP explanations of feature importance.
- **Tabular.ipynb** – Adult Income prediction on the UCI Census dataset, covering EDA, preprocessing, classical ML models, an ANN baseline, dimensionality reduction, and clustering analyses.
- **Textual.ipynb** – Fake news detection that combines classical NLP models and a transformer baseline, plus topic modeling via K-Means over vectorized text features.

## Environment Setup
1. Create and activate a Python 3.9+ virtual environment.
2. Install core dependencies:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn tensorflow torch torchvision torchaudio transformers nltk shap librosa kagglehub notebook
   ```
3. Download NLTK resources required by the text notebook (e.g., `punkt`, `stopwords`) ahead of time to avoid runtime downloads:
   ```bash
   python -m nltk.downloader punkt stopwords averaged_perceptron_tagger wordnet
   ```

## Data Requirements
- **Image.ipynb:** Expects an image dataset in `PV/`, organized into subdirectories per class. Update `FOLDER_PATH` in the notebook if your images are stored elsewhere.
- **Sound.ipynb:** Uses the Kaggle `chrisfilo/urbansound8k` dataset fetched via `kagglehub`. Ensure your Kaggle API token is available (see below) or place the UrbanSound8K files locally and point the `path` variable accordingly.
- **Tabular.ipynb:** Requires `adult.csv` and `adult_test.csv` from the UCI Adult Income dataset in the repository root (or adjust the file paths in the loading cell).
- **Textual.ipynb:** Downloads the Kaggle `clmentbisaillon/fake-and-real-news-dataset` via `kagglehub`. Provide Kaggle credentials or replace the download step with a local path to the dataset.

- ## Running the Notebooks
1. Open the desired notebook and run cells sequentially. Some training sections are computationally intensive; consider enabling GPU acceleration when available.

## Notes
- Random seeds are set in the notebooks for reproducibility, but results may still vary depending on library versions and hardware.
- Adjust batch sizes, epochs, and model choices in the notebooks to fit your compute budget.
