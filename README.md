# lensword
Deep learning Sentiment analysis for e-commerce product review
# LensWord 🔍

### Deep Learning Sentiment Analysis for E-Commerce Product Reviews

> **Bidirectional LSTM + RAG-Powered Response System**  
> Built by Betty George & Miheret Woldegabrial — AI/ML Engineering Program, 2026

---

## What is LensWord?

LensWord is an end-to-end deep learning system that automatically classifies Amazon product reviews as **Positive**, **Neutral**, or **Negative** — and uses a RAG (Retrieval-Augmented Generation) system to suggest the most relevant customer service response based on the complaint.

Instead of a business reading thousands of reviews manually, LensWord does it instantly and consistently, helping teams prioritize customer issues and respond faster.

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

## Architecture

```
User Review (index.html)
        ↓
FastAPI /predict endpoint (api.py)
        ↓
Text Preprocessing (clean → tokenize → pad)
        ↓
Bidirectional LSTM (2 layers, 872,451 parameters)
        ↓
Sentiment Prediction (Negative / Neutral / Positive)
        ↓
RAG System (ChromaDB + SentenceTransformers)
        ↓
Suggested Customer Service Response
        ↓
JSON Response → index.html displays results
```

---

## Project Structure

```
lensword/
├── data/
│   ├── amazon_reviews.csv              # Original Amazon Alexa dataset
│   ├── amazon_reviews_cleaned.csv      # Combined Amazon + Yelp dataset (9,149 reviews)
│   ├── word2idx.pkl                    # Vocabulary dictionary
│   └── *.pt                           # Preprocessed PyTorch tensors
├── models/
│   ├── lensword_model.pt              # Trained LSTM model weights
│   ├── training_curves.png            # Loss and accuracy curves
│   ├── confusion_matrix.png           # Confusion matrix
│   ├── final_model_comparison.png     # Model comparison chart
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
│   └── knowledge_base.csv            # RAG customer service knowledge base
├── reports/                           # Technical report and presentation slides
├── Dockerfile                         # Docker containerization
├── .dockerignore
├── index.html                         # JavaScript web app frontend
└── requirements.txt                   # Python dependencies
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Deep Learning | PyTorch — Bidirectional LSTM |
| Text Processing | Custom tokenizer, SMOTE (imbalanced-learn) |
| Data | Amazon Alexa Reviews + Yelp Reviews (HuggingFace) |
| RAG | ChromaDB + SentenceTransformers (all-MiniLM-L6-v2) |
| API | FastAPI + Uvicorn |
| Frontend | HTML + CSS + JavaScript |
| Containerization | Docker |
| Version Control | GitHub |

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
- **Web App:** Open `index.html` in your browser
- **API Health Check:** http://127.0.0.1:8000
- **API Docs:** http://127.0.0.1:8000/docs

---

## Usage

1. Open `index.html` in your browser
2. Type or paste a product review
3. Click **Analyze Sentiment**
4. LensWord returns:
   - Sentiment classification (Positive / Neutral / Negative)
   - Confidence score
   - Probability breakdown for all three classes
   - RAG-powered suggested customer service response
   - Priority level (LOW / MEDIUM / HIGH)

### Example API Request
```json
POST http://127.0.0.1:8000/predict
{
  "text": "This product broke after two days, terrible quality"
}
```

### Example API Response
```json
{
  "text": "This product broke after two days, terrible quality",
  "sentiment": "Negative",
  "confidence": 87.32,
  "probabilities": {
    "Negative": 87.32,
    "Neutral": 5.41,
    "Positive": 7.27
  },
  "action": "escalate",
  "priority": "HIGH",
  "suggested_response": "We sincerely apologize for the quality issue you experienced. We would like to offer you a full replacement or refund. Please contact our support team with your order number and we will resolve this immediately."
}
```

---

## Model Details

### Architecture
- **Embedding Layer:** 4,340 vocabulary → 64 dimensions
- **Bidirectional LSTM:** 2 layers, 128 hidden units, reads forward and backward
- **Dropout:** 0.3 regularization
- **Fully Connected:** 256 → 3 classes
- **Total Parameters:** 872,451

### Training
- **Dataset:** 9,149 reviews (Amazon Alexa + Yelp)
- **SMOTE:** Applied to balance classes to 3,555 each → 10,665 training samples
- **Optimizer:** Adam (lr=0.001)
- **Scheduler:** ReduceLROnPlateau (patience=2, factor=0.5)
- **Early Stopping:** Patience=5
- **Best Validation Accuracy:** 89.61%

### Key Experiments
| Experiment | Macro F1 | Finding |
|---|---|---|
| Baseline LSTM | 67.99% | Heavy class imbalance |
| + SMOTE | 72.42% | Improved minority classes |
| + LR Scheduler | 76.49% | Best single run |
| + Expanded Dataset | **88.40%** | Addressed advisor feedback |

---

## Academic Integrity

This project was completed as part of the AI/ML Engineering Program at Apeiron AI Training. Claude AI was used as a learning assistant for guidance, debugging, and concept explanation. All code was written, understood, and executed by the team. Any significant AI assistance has been disclosed in accordance with program guidelines.

---

## Team

**Betty George** — Co-Lead, AI/ML Engineering Program  
**Miheret Woldegabrial** — Co-Lead, AI/ML Engineering Program