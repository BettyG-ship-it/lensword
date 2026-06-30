# LensWord - FastAPI Sentiment Prediction API (Ensemble Version)
# Uses 3 models trained with different seeds, averaging their predictions
# for more stable and reproducible results

from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import pickle
import re
import os
import sys

# Add project root to path so we can import config.py
sys.path.append(os.path.abspath('..'))
from src.config import *

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="LensWord Sentiment Analysis API (Ensemble)")

# -----------------------------
# Request model
# -----------------------------
class ReviewRequest(BaseModel):
    text: str

# -----------------------------
# Model architecture
# -----------------------------
class LensWordLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers,
                 num_classes, dropout):
        super(LensWordLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        return output

# -----------------------------
# Load vocabulary
# -----------------------------
with open('../data/word2idx.pkl', 'rb') as f:
    word2idx = pickle.load(f)

# -----------------------------
# Load ensemble models
# -----------------------------
SEEDS = [42, 123, 7]
ensemble_models = []

for seed in SEEDS:
    m = LensWordLSTM(
        vocab_size=MAX_VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT
    )
    m.load_state_dict(torch.load(f'../models/lensword_model_seed{seed}.pt'))
    m.eval()
    ensemble_models.append(m)

print(f"Loaded {len(ensemble_models)} ensemble models successfully!")

# -----------------------------
# Root endpoint
# -----------------------------
@app.get("/")
def root():
    return {
        "message": "LensWord Sentiment Analysis API is running!",
        "model_type": "Ensemble (3 models)",
        "seeds": SEEDS
    }

# -----------------------------
# Helper functions
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

def text_to_sequence(text):
    return [word2idx.get(word, word2idx["<UNK>"]) for word in text.split()]

def pad_sequence(seq, max_len):
    if len(seq) < max_len:
        return seq + [0] * (max_len - len(seq))
    return seq[:max_len]

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict_sentiment(request: ReviewRequest):

    # Step 1 - Clean text
    cleaned = clean_text(request.text)

    # Step 2 - Convert to sequence
    sequence = text_to_sequence(cleaned)

    # Step 3 - Pad
    padded = pad_sequence(sequence, MAX_SEQ_LENGTH)

    # Step 4 - Tensor
    input_tensor = torch.tensor([padded], dtype=torch.long)

    # Step 5 - Ensemble predictions
    all_probs = []
    with torch.no_grad():
        for m in ensemble_models:
            output = m(input_tensor)
            probs = torch.softmax(output, dim=1)
            all_probs.append(probs)

    avg_probs = torch.stack(all_probs).mean(dim=0)
    predicted_class = avg_probs.argmax(dim=1).item()
    confidence = avg_probs[0][predicted_class].item()

    # Step 6 - Map label
    labels = ['Negative', 'Neutral', 'Positive']
    sentiment = labels[predicted_class]

    return {
        "text": request.text,
        "sentiment": sentiment,
        "confidence": round(confidence * 100, 2),
        "model_type": "ensemble_3_models"
    }
