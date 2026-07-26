# Hate Speech Identifier

This project provides a Python 3.12 scaffold for a TensorFlow-based NLP system
for hate-speech detection.

## Project Structure

- datasets/raw/davidson/ - Raw Davidson dataset files
- datasets/raw/hatexplain/ - Raw HateXplain dataset files
- datasets/processed/ - Processed datasets
- datasets/splits/ - Train/validation/test split artifacts
- models/ - Trained model files and checkpoints
- scripts/ - Utility and preprocessing scripts

## Environment

- Python 3.12
- Follow PEP 8 style guidelines
- Type hints are used throughout the Python code
- Comprehensive docstrings are included in the starter modules

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the training entry point:
   ```bash
   python train_model.py
   ```
4. Launch the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

## Notes

The current repository contains boilerplate files only. Model implementation
will be added in future steps.
