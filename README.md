# ✈️ From Passenger Voice to Operational Intelligence
## A Transformer-Based NLP System for Airline Experience Analysis

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF)
![License](https://img.shields.io/badge/License-MIT-green)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF)
![License](https://img.shields.io/badge/License-MIT-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-189A3A)
![LightGBM](https://img.shields.io/badge/LightGBM-4.3%2B-02569B)
![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-154f3c)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458)
![NumPy](https://img.shields.io/badge/NumPy-1.26%2B-013243)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8%2B-11557C)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13%2B-4C72B0)
![RoBERTa](https://img.shields.io/badge/RoBERTa-cardiffnlp-red)
![DistilBERT](https://img.shields.io/badge/DistilBERT-base--uncased-orange)
![BART](https://img.shields.io/badge/BART-large--cnn-purple)
![Datasets](https://img.shields.io/badge/HuggingFace-Datasets-FFD21E)
![Accelerate](https://img.shields.io/badge/HuggingFace-Accelerate-FFD21E)
![FIAP](https://img.shields.io/badge/FIAP-Data%20Science%20%26%20AI-red)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![NLP](https://img.shields.io/badge/Task-NLP%20Sentiment%20Analysis-blueviolet)
![Sentiment](https://img.shields.io/badge/Classes-Negative%20%7C%20Neutral%20%7C%20Positive-informational)

![](https://github.com/RafaelGallo/NLP_Transformer_Airline_Reviews/blob/main/imgs/log.png?raw=true)

## Business Problem

The global aviation industry processes millions of passenger reviews annually
across dozens of carriers, routes, cabin classes, and aircraft types. While
structured rating fields provide a quantifiable snapshot of the passenger
experience, the most actionable intelligence remains locked inside free-text
reviews — unstructured, high-volume, and impossible to analyze at scale through
manual processes.

This project proposes a **Transformer-based NLP intelligence system** applied to
the Global Airline Reviews dataset, designed to serve a customer experience
consultancy delivering automated, scalable insights to airline operations, product,
and marketing teams.

## Dataset

**Global Airline Reviews — A-Z Airline Feedback**

| Column | Description |
|---|---|
| NAMES | Airline name |
| Review | Full-text passenger review |
| clean_text | Preprocessed review text |
| Aircraft | Aircraft model flown |
| Seat Type | Cabin class (Economy, Business, First) |
| Type Of Traveller | Solo, Couple, Business, Family |
| Route | Flight origin and destination |
| Seat Comfort | Rating 1–5 |
| Cabin Staff Service | Rating 1–5 |
| Food & Beverages | Rating 1–5 |
| Inflight Entertainment | Rating 1–5 |
| Ground Service | Rating 1–5 |
| Wifi & Connectivity | Rating 1–5 |
| Value For Money | Rating 1–5 |
| Recommended | Yes / No |
| sentiment | Negative / Neutral / Positive |
| sentiment_encoded | 0 / 1 / 2 |

## Project Structure

```
├── data/
│   └── airline_reviews.csv
├── models/
│   ├── roberta_sentiment/
│   ├── bert_sentiment/
│   ├── tfidf_vectorizer.pkl
│   ├── label_encoder.pkl
│   ├── lr_bert_embeddings.pkl
│   └── bart_sentiment_summaries.csv
├── notebooks/
│   └── airline_reviews_nlp.ipynb
├── requirements.txt
└── README.md
```

## Modeling Tasks

### Task 1 — Sentiment Classification (RoBERTa / DistilBERT)
Fine-tuned Transformer models to classify passenger reviews into
Negative, Neutral, or Positive sentiment using `clean_text` as input
and `sentiment_encoded` as target.

### Task 2 — Classical ML Baseline (TF-IDF)
Ten classical ML models trained on TF-IDF features as baseline
comparison for Transformer models.

### Task 3 — Abstractive Summarization (BART)
Zero-shot summarization pipeline using `facebook/bart-large-cnn` to
generate executive briefings grouped by sentiment class.

### Task 4 — Embedding-Based Inference
DistilBERT CLS token embeddings extracted and used as input features
for a Logistic Regression classifier — zero-shot transfer evaluation.

## Results

### Transformer Models

| Rank | Model | Accuracy | F1 Weighted | F1 Macro | AUC |
|---|---|---|---|---|---|
| 🥇 | RoBERTa | 0.8969 | 0.9004 | 0.6787 | 0.9563 |
| 🥈 | DistilBERT | 0.8807 | 0.8790 | 0.6280 | 0.9468 |
| 🥉 | LR + BERT Embeddings | 0.8571 | 0.8581 | 0.5988 | — |

## Results

### Final Model Comparison — All Strategies

| Rank | Model | Strategy | Accuracy | F1 Weighted | F1 Macro | AUC |
|---|---|---|---|---|---|---|
| 🥇 | RoBERTa | Transformer Fine-Tuning | 0.8969 | 0.9004 | 0.6787 | 0.9563 |
| 🥈 | DistilBERT | Transformer Fine-Tuning | 0.8807 | 0.8790 | 0.6280 | 0.9468 |
| 🥉 | LR + BERT Embeddings | Embedding-Based | 0.8571 | 0.8581 | 0.5988 | — |
| 4 | XGBoost | Classical ML (TF-IDF) | 0.8716 | 0.8642 | 0.6210 | — |
| 5 | LightGBM | Classical ML (TF-IDF) | 0.8662 | 0.8591 | 0.6170 | — |
| 6 | SGDClassifier | Classical ML (TF-IDF) | 0.8608 | 0.8525 | 0.6111 | — |
| 7 | LinearSVC | Classical ML (TF-IDF) | 0.8571 | 0.8479 | 0.6064 | — |
| 8 | LogisticRegression | Classical ML (TF-IDF) | 0.8228 | 0.8004 | 0.5067 | — |
| 9 | RandomForest | Classical ML (TF-IDF) | 0.8174 | 0.7996 | 0.5601 | — |
| 10 | DecisionTree | Classical ML (TF-IDF) | 0.7776 | 0.7608 | 0.5190 | — |
| 11 | BernoulliNB | Classical ML (TF-IDF) | 0.7161 | 0.6080 | 0.3026 | — |
| 12 | MultinomialNB | Classical ML (TF-IDF) | 0.7089 | 0.5917 | 0.2849 | — |
| 13 | KNN | Classical ML (TF-IDF) | 0.7071 | 0.5876 | 0.2803 | — |

### Key Observations

**Transformers vs Classical ML**
RoBERTa outperformed all classical ML models across every metric, confirming
that contextual representations capture semantic relationships that
frequency-based TF-IDF approaches cannot resolve — particularly negation,
contrast, and irony patterns common in airline reviews.

**XGBoost and LightGBM as strong baselines**
Among classical ML models, XGBoost (F1=0.8642) and LightGBM (F1=0.8591)
delivered the strongest results, outperforming even LR + BERT Embeddings
(F1=0.8581) — a notable finding that highlights the competitiveness of
gradient boosting on TF-IDF features for this domain.

**SGDClassifier and LinearSVC as efficient alternatives**
Both models achieved F1 above 0.85 with significantly lower training cost
than Transformer fine-tuning — viable options for latency-sensitive
production environments.

**Naive Bayes and KNN underperformed**
BernoulliNB, MultinomialNB, and KNN all fell below 0.72 in F1 Weighted,
confirming their limitations in capturing the semantic complexity of
passenger reviews with high-dimensional TF-IDF sparse features.

**Neutral class remains the consistent weak point**
All models struggled with the Neutral class (50 samples), regardless of
strategy. This is a data volume constraint rather than a modeling limitation,
and requires targeted data collection or augmentation before production
deployment.

## Pipeline Architecture

```
Raw Reviews (free text)
        │
        ▼
┌─────────────────────┐
│   Preprocessing     │  emoji removal, cleaning, lemmatization
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
[TF-IDF]     [Tokenizer]
Classical    Transformer
ML Models    Models
    │             │
    ▼             ▼
Logistic     RoBERTa /
Regression   DistilBERT
SVM, XGB     Fine-tuning
LightGBM         │
    │             ▼
    └──────┬──────┘
           ▼
  Sentiment Prediction
  (Negative/Neutral/Positive)
           │
           ▼
  ┌────────────────┐
  │  BART Zero-Shot │
  │  Summarization  │
  └────────────────┘
           │
           ▼
  Executive Briefing
  by Sentiment Group
```

## Dimensionality Reduction

| Technique | Input | Purpose |
|---|---|---|
| PCA | TF-IDF dense matrix | Linear separability analysis |
| t-SNE | PCA output | Non-linear cluster visualization |

## Class Imbalance Handling

The dataset presented significant class imbalance:

| Class | Samples | Weight |
|---|---|---|
| Negative | 765 | 1.2021 |
| Neutral | 50 | 18.3981 |
| Positive | 1946 | 0.4731 |

Strategy applied: `WeightedTrainer` with `CrossEntropyLoss` weighted
by `compute_class_weight(class_weight='balanced')` from scikit-learn.

## Installation

```bash
pip install transformers datasets accelerate sentencepiece
pip install scikit-learn xgboost lightgbm imbalanced-learn
pip install pandas numpy matplotlib seaborn tqdm joblib emoji nltk
```

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "models/roberta_sentiment"
tokenizer  = AutoTokenizer.from_pretrained(model_path)
model      = AutoModelForSequenceClassification.from_pretrained(model_path)

text     = "The cabin crew was fantastic and the seats were very comfortable."
inputs   = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
logits   = model(**inputs).logits
pred     = torch.argmax(logits, dim=-1).item()

labels   = {0: "Negative", 1: "Neutral", 2: "Positive"}
print(f"Prediction: {labels[pred]}")
```

## Key Libraries

| Library | Version | Purpose |
|---|---|---|
| transformers | 4.40+ | RoBERTa, DistilBERT, BART |
| datasets | 2.18+ | HuggingFace Dataset pipeline |
| scikit-learn | 1.4+ | Classical ML, metrics, PCA, t-SNE |
| xgboost | 2.0+ | Gradient boosting baseline |
| lightgbm | 4.3+ | Gradient boosting baseline |
| torch | 2.2+ | Transformer training backend |
| nltk | 3.8+ | Tokenization, stopwords, lemmatization |
| emoji | 2.11+ | Emoji removal from reviews |

## Limitations

- **Neutral class** — only 50 samples limits generalization across all models.
  Next step: data augmentation via back-translation or paraphrasing with T5.
- **Monolingual** — pipeline trained on English reviews only. Extension to
  multilingual with `xlm-roberta-base` is planned.
- **BART zero-shot** — summarization quality would improve significantly with
  fine-tuning on aviation-specific (review, summary) pairs.

## Next Steps

- Data augmentation for the Neutral class (back-translation, T5 paraphrasing)
- Multilingual extension with `xlm-roberta-base`
- BART fine-tuning on aviation domain summaries
- Production deployment via FastAPI + MLflow + Airflow on Oracle Cloud Free Tier
- Real-time inference API with model versioning and automated retraining

