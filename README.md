# LensWord 🔍

### Deep Learning Sentiment Analysis for E-Commerce Product Reviews

> **Bidirectional LSTM + RAG-Powered Response System**  
> Built by Betty George & Miheret Woldegabrial — AI/ML Engineering Program, 2026

---

## What is LensWord?

LensWord is an end-to-end deep learning system that automatically classifies Amazon product reviews as **Positive**, **Neutral**, or **Negative** — and uses a RAG (Retrieval-Augmented Generation) system to suggest the most relevant customer service response.

A star rating tells you HOW unhappy a customer is — LensWord tells you **WHY**, **how urgent it is**, and **exactly what to say back**.

---

## Honest Results (Clean Pipeline)

| Model | Accuracy | Macro F1 |
|---|---|---|
| **LensWord LSTM (trained, in-domain)** | **72.62%** | **0.7263** |
| NLPTown (zero-shot) | 72.52% | 0.7115 |
| LiYuan Amazon (zero-shot) | 64.47% | 0.6241 |
| CardiffNLP Twitter (zero-shot) | 60.68% | 0.5399 |

> **Note:** LensWord was trained on this distribution. HuggingFace models are evaluated zero-shot — they were never shown our data or label rules. The comparison demonstrates domain-specific training value. All models evaluated on the same 1,030 held-out test rows.

### Per-Class F1 Scores

| Class | F1 Score |
|---|---|
| Negative | 0.7467 |
| Neutral | 0.6396 |
| Positive | 0.7925 |
| **Macro F1** | **0.7263** |

---

## Why Numbers Changed

Earlier runs reported 88.85% accuracy and 88.40% Macro F1. Following a full pipeline audit these figures were found to be inflated due to:

- **Duplicate leakage** — ~6,000 duplicate rows distributed across train and test
- **Vocabulary leakage** — vocabulary fitted on all data including test partition  
- **Wrong checkpoint** — model selected on accuracy not Macro F1
- **Invalid SMOTE** — applied to token-index sequences (nominal feature space)

The corrected figures represent honest performance on genuinely unseen, uncontaminated data.

---

## Dual Interface System

### 1. Business Dashboard — `index.html`
For internal business and customer service teams.

![Business Dashboard](screenshots/dashboard.png)

**Features:**
- Real-time sentiment classification with color-coded badge
- Confidence score and probability breakdown for all three classes
- Priority level (LOW / MEDIUM / HIGH) from API
- RAG-powered suggested customer service response from API

### 2. Customer Chat Interface — `customer.html`
For direct customer interaction after submitting a review.

![Customer Chat](screenshots/customer_chat.png)

**Features:**
- Customer submits review naturally
- System detects sentiment silently — customer never sees technical outputs
- 5-step empathetic conversation flow based on detected sentiment
- 75% confidence threshold — low confidence predictions ask customer directly
- **Negative** → empathetic resolution with replacement/refund options
- **Neutral** → feedback collection with follow-up questions
- **Positive** → warm appreciation with loyalty program offer

---

## Architecture

```
Review text
        ↓
Tokenization → word2idx vocabulary (4,340 words, fitted on training only)
        ↓
Padding to 50 tokens
        ↓
Embedding Layer [50] → [50, 64]
        ↓
BiLSTM Layer 1 (forward + backward) → [50, 128]
        ↓
BiLSTM Layer 2 → final hidden state [128]
        ↓
Dropout (p=0.4)
        ↓
Fully Connected → [3 logits]
        ↓
Softmax → [3 probabilities]
        ↓
Sentiment (Negative / Neutral / Positive)
        ↓
RAG System (SentenceTransformer + ChromaDB, 33 entries)
        ↓
Suggested Customer Service Response
        ↓
index.html (business) or customer.html (customer chat)
```

---

## Project Structure

```
lensword/
├── data/
│   ├── amazon_reviews_cleaned.csv      # Notebook 01 output — READ ONLY
│   ├── amazon_yelp_combined.csv        # Notebook 02 output
│   ├── test_texts.csv                  # Test rows saved at split time (provenance)
│   ├── word2idx.pkl                    # Vocabulary (fitted on training only)
│   └── *.pt                           # PyTorch tensors
├── models/
│   ├── lensword_model.pt              # Trained LSTM weights
│   ├── metrics.json                   # Official results (all figures from here)
│   ├── comparison_results.json        # HuggingFace comparison results
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   └── final_model_comparison.png
├── notebooks/
│   ├── 01_EDA_lensword.ipynb
│   ├── 02_preprocessing_lensword.ipynb
│   ├── 03_model_training_lensword.ipynb
│   ├── 04_evaluation_lensword.ipynb
│   └── 05_huggingface_comparison_lensword.ipynb
├── src/
│   ├── api.py                         # FastAPI + RAG endpoint
│   ├── config.py                      # Hyperparameters
│   └── knowledge_base.csv            # 33-entry RAG knowledge base
├── screenshots/
├── reports/
├── Dockerfile
├── index.html                         # Business dashboard
├── customer.html                      # Customer chat interface
└── requirements.txt
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Deep Learning | PyTorch — Bidirectional LSTM |
| Text Processing | Custom tokenizer, inverse-frequency class weights |
| Data | Amazon Alexa Reviews (Kaggle) + Yelp Reviews (HuggingFace) |
| RAG | ChromaDB + SentenceTransformers (all-MiniLM-L6-v2) |
| API | FastAPI + Uvicorn |
| Frontend | HTML + CSS + JavaScript |
| Containerization | Docker |
| Version Control | GitHub |

---

## HuggingFace Resources Used

| Resource | Purpose |
|---|---|
| `Yelp/yelp_review_full` | 8,000 additional training reviews |
| `nlptown/bert-base-multilingual-uncased-sentiment` | Zero-shot comparison baseline |
| `LiYuan/amazon-review-sentiment-analysis` | Zero-shot comparison baseline |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | Zero-shot comparison baseline |
| `sentence-transformers/all-MiniLM-L6-v2` | RAG embeddings (384-dim vectors) |

---

## Setup Instructions

### Prerequisites
- Python 3.11+
- Git

### 1. Clone the repository
```bash
git clone https://github.com/BettyG-ship-it/lensword.git
cd lensword
```

### 2. Create and activate virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run notebooks in order
```
01_EDA_lensword.ipynb
02_preprocessing_lensword.ipynb
03_model_training_lensword.ipynb
04_evaluation_lensword.ipynb
05_huggingface_comparison_lensword.ipynb
```

### 5. Start the API
```bash
cd src
uvicorn api:app --reload
```

### 6. Open the interfaces
- **Business Dashboard:** Open `index.html` in your browser
- **Customer Chat:** Open `customer.html` in your browser
- **API Docs:** http://127.0.0.1:8000/docs

---

## Docker
```bash
docker build -t lensword .
docker run -p 8000:8000 lensword
```

---

## Model Details

| Parameter | Value |
|---|---|
| Architecture | Bidirectional LSTM |
| Layers | 2 |
| Hidden Dimensions | 64 |
| Embedding Dimensions | 64 |
| Vocabulary Size | 4,340 (fitted on training only) |
| Max Sequence Length | 50 tokens |
| Total Parameters | 444,035 |
| Optimizer | AdamW (weight_decay=1e-4) |
| Dropout | 0.4 |
| Test Accuracy | 72.62% |
| Test Macro F1 | 0.7263 |

---

## Known Limitations

- Overfitting present — ~10% gap between training and validation accuracy due to limited dataset size
- Neutral class is hardest to classify (F1: 0.6396) — 3-star reviews are genuinely ambiguous
- English language only
- RAG knowledge base contains 33 entries

---

## Academic Integrity

This project was completed as part of the AI/ML Engineering Program at Apeiron AI Training. Claude AI was used as a learning assistant for guidance, debugging, and concept explanation. All code was written, understood, and executed by the team.

---

## Team

**Betty George** — Co-Lead, AI/ML Engineering Program  
**Miheret Woldegabrial** — Co-Lead, AI/ML Engineering Program