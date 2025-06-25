# English-to-French Translation Model

This notebook demonstrates the process of fine-tuning a sequence-to-sequence model for English-to-French translation using the Helsinki-NLP/opus-mt-en-fr model and the KDE4 dataset. The trained model is deployed as a web application using Streamlit.

## Overview

The project includes:
- Loading and preprocessing the KDE4 parallel corpus (English-French)
- Fine-tuning the OPUS-MT model for translation tasks
- Evaluating model performance using SacreBLEU metric
- Saving the trained model for future use
- Deployment as a web application using Streamlit

## Streamlit Demo
![image](https://github.com/user-attachments/assets/f355512d-2467-409b-b5b1-5f1ebe5da591)

Watch the demo video:  
[![Video Demonstration](https://img.shields.io/badge/▶-Watch%20Demo-red.svg)](https://drive.google.com/file/d/1nc14-nnnqE2sknREWyQA2IStaoBAWoQ7/view?usp=sharing)

## Key Features

- Utilizes Hugging Face's Transformers library for model training
- Implements efficient data loading and preprocessing pipelines
- Includes evaluation with standard machine translation metrics
- Supports GPU acceleration for faster training
- Simple web interface for real-time translation

## Requirements

To run this notebook, you'll need:

- Python 3.11+
- PyTorch with CUDA support
- Hugging Face Transformers
- Datasets library
- SacreBLEU for evaluation
- Accelerate for distributed training
- Streamlit for deployment

Install requirements with:
```bash
pip install transformers datasets sacrebleu evaluate accelerate streamlit
```

## Dataset

The model is trained on the KDE4 localization dataset, which contains:
- 210,173 parallel English-French sentence pairs
- Technical and UI-related translations from KDE software

## Model Architecture

The fine-tuned model uses:
- MarianMT architecture (Transformer-based)
- Pretrained Helsinki-NLP/opus-mt-en-fr weights
- Sequence-to-sequence learning objective

## Training Configuration

- Batch size: 32 (training), 64 (evaluation)
- Learning rate: 2e-5
- Weight decay: 0.01
- Maximum sequence length: 128 tokens
- Training epochs: 3
- Mixed precision training (FP16)

## Evaluation Metrics

Model performance is measured using SacreBLEU score, achieving 46.92% BLEU on the validation set.

## Deployment

The model is deployed as a web application using Streamlit, providing:
- Real-time English-to-French translation
- Clean, user-friendly interface
- Responsive design for various devices

To run the Streamlit app locally:
```bash
streamlit run app.py
```
