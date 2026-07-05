# LensWord 🔍

### Deep Learning Sentiment Analysis for E-Commerce Product Reviews

> **Bidirectional LSTM + RAG-Powered Response System**  
> Built by Betty George & Miheret Woldegabrial — AI/ML Engineering Program, 2026

---

## What is LensWord?

LensWord is an end-to-end deep learning system that automatically classifies Amazon product reviews as **Positive**, **Neutral**, or **Negative** — and uses a RAG (Retrieval-Augmented Generation) system to suggest the most relevant customer service response.

A star rating tells you HOW unhappy a customer is — LensWord tells you **WHY**, **how urgent it is**, and **exactly what to say back**.

---

## Key Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| **LensWord LSTM (Final)** | **88.85%** | **88.40%** |
| HuggingFace NLPTown | 73.00% | 67.89% |
| HuggingFace LiYuan (Amazon) | 70.00% | 63.86% |
| HuggingFace CardiffNLP | 67.00% | 56.70% |

> LensWord outperforms all three pretrained HuggingFace models on both accuracy and Macro F1.

---

## Dual Interface System

LensWord provides two separate interfaces serving two different audiences:

### 1. Business Dashboard — `index.html`
For internal business and customer service teams.

![Business Dashboard](screenshots/dashboard.png)

**Features:**
- Real-time sentiment classification with color-coded badge
- Confidence score and probability breakdown for all three classes
- Priority level (LOW / MEDIUM / HIGH)
- RAG-powered suggested customer service response

### 2. Customer Chat Interface — `customer.html`
For direct customer interaction after submitting a review.

![Customer Chat](screenshots/customer_chat.png)

**Features:**
- Customer submits their review naturally
- System detects sentiment silently — customer never sees technical outputs
- 5-step empathetic conversation flow based on detected sentiment:
  - **Negative** → empathetic resolution with replacement/refund options
  - **Neutral** → feedback collection with follow-up questions
  - **Positive** → warm appreciation with loyalty program offer
- Confidence threshold: if model confidence below 75%, asks customer directly

---

## Architecture

```
User Review
        ↓
FastAPI /predict endpoint (api.py)
        ↓
Text Preprocessing (clean → tokenize → pad to 50 tokens)
        ↓
Bidirectional LSTM (2 layers, 872,451 parameters)
        ↓
Sentiment Prediction (Negative / Neutral / Positive)
        ↓
RAG System (ChromaDB + SentenceTransformers all-MiniLM-L6-v2)
        ↓
Suggested Customer Service Response
        ↓
index.html (business dashboard) or customer.html (chat interface)
```

---

## Project Structure

```
lensword/
├── data/
│   ├── amazon_reviews.csv              # Original Amazon Alexa dataset (Kaggle)
│   ├── amazon_reviews_cleaned.csv      # Combined Amazon + Yelp dataset (9,149 reviews)
│   ├── word2idx.pkl                    # Vocabulary dictionary (4,340 words)
│   └── *.pt                           # Preprocessed PyTorch tensors
├── models/
│   ├── lensword_model.pt              # Trained LSTM model weights
│   ├── training_curves.png            # Loss and accuracy curves
│   ├── confusion_matrix.png           # Confusion matrix
│   └── results_comparison.csv        # Full results table
├── notebooks/
│   ├── 01_EDA_lensword.ipynb          # Exploratory Data Analysis
│   ├── 02_preprocessing_lensword.ipynb # Text preprocessing + SMOTE
│   ├── 03_model_training_lensword.ipynb # LSTM training
│   ├── 04_evaluation_lensword.ipynb    # Model evaluation
│   └── 05_huggingface_comparison_lensword.ipynb # HuggingFace comparison
├── src/
│   ├── api.py                         # FastAPI + RAG prediction endpoint
│   ├── config.py                      # Project configuration
│   └── knowledge_base.csv            # RAG customer service knowledge base (14 entries)
├── reports/                           # Technical report and presentation slides
├── screenshots/                       # Interface screenshots
│   ├── dashboard.png                  # Business dashboard screenshot
│   └── customer_chat.png             # Customer chat interface screenshot
├── Dockerfile                         # Docker containerization
├── index.html                         # Business analytics dashboard
├── customer.html                      # Customer conversational chatbot
└── requirements.txt                   # Python dependencies
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Deep Learning | PyTorch — Bidirectional LSTM |
| Text Processing | Custom tokenizer, SMOTE (imbalanced-learn) |
| Data | Amazon Alexa Reviews (Kaggle) + Yelp Reviews (HuggingFace) |
| RAG | ChromaDB + SentenceTransformers (all-MiniLM-L6-v2) |
| API | FastAPI + Uvicorn |
| Frontend | HTML + CSS + JavaScript (no frameworks) |
| Containerization | Docker |
| GPU Training | Google Colab T4 GPU |
| Version Control | GitHub |

---

## HuggingFace Resources Used

| Resource | Purpose |
|---|---|
| `Yelp/yelp_review_full` | Additional training data (6,000 reviews) |
| `nlptown/bert-base-multilingual-uncased-sentiment` | Comparison baseline |
| `LiYuan/amazon-review-sentiment-analysis` | Comparison baseline |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | Comparison baseline |
| `sentence-transformers/all-MiniLM-L6-v2` | RAG embeddings (384-dim vectors) |

---

## Setup Instructions

### Prerequisites
- Python 3.11+
- Git
- Docker Desktop (optional)

### 1. Clone the repository
```bash
git clone https://github.com/BettyG-ship-it/lensword.git
cd lensword
```

### 2. Create and activate virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install "fastapi[all]"
pip install chromadb sentence-transformers
```

### 4. Download the dataset
Download `amazon_reviews.csv` from [Kaggle — Amazon Alexa Reviews](https://www.kaggle.com/datasets/sid321axn/amazon-alexa-reviews) and place it in the `data/` folder.

### 5. Run the notebooks in order
```
01_EDA_lensword.ipynb
02_preprocessing_lensword.ipynb
03_model_training_lensword.ipynb
04_evaluation_lensword.ipynb
05_huggingface_comparison_lensword.ipynb
```

---

## Running the Application

### Option A — Run with FastAPI directly
```bash
cd src
uvicorn api:app --reload
```

### Option B — Run with Docker
```bash
docker build -t lensword .
docker run -p 8000:8000 lensword
```

### Access the application
- **Business Dashboard:** Open `index.html` in your browser
- **Customer Chat:** Open `customer.html` in your browser
- **API Health Check:** http://127.0.0.1:8000
- **API Docs:** http://127.0.0.1:8000/docs

---

## Example API Request & Response

```json
POST http://127.0.0.1:8000/predict
{
  "text": "This product broke after two days, terrible quality"
}
```

```json
{
  "sentiment": "Negative",
  "confidence": 87.32,
  "probabilities": {
    "Negative": 87.32,
    "Neutral": 5.41,
    "Positive": 7.27
  },
  "action": "escalate",
  "priority": "HIGH",
  "suggested_response": "We sincerely apologize for the quality issue. We would like to offer you a full replacement or refund."
}
```

---

## Model Details

| Parameter | Value |
|---|---|
| Architecture | Bidirectional LSTM |
| Layers | 2 |
| Hidden Dimensions | 128 |
| Embedding Dimensions | 64 |
| Vocabulary Size | 4,340 |
| Max Sequence Length | 50 tokens |
| Total Parameters | 872,451 |
| Best Validation Accuracy | 89.61% |
| Macro F1 Score | 88.40% |

---

## Known Limitations

- Mild overfitting (10% gap between training and validation accuracy) due to limited dataset size
- Negation handling weakness — addressed via confidence threshold in customer.html
- English language only
- RAG knowledge base contains 14 entries

---

## Academic Integrity

This project was completed as part of the AI/ML Engineering Program at Apeiron AI Training. Claude AI was used as a learning assistant for guidance, debugging, and concept explanation. All code was written, understood, and executed by the team.

---

## Team

**Betty George** — Co-Lead, AI/ML Engineering Program  
**Miheret Woldegabrial** — Co-Lead, AI/ML Engineering Program