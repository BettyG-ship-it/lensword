# LensWord - FastAPI Sentiment Prediction API (Ensemble Version)
# Uses 3 models trained with different seeds for stable predictions

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

# Create the FastAPI app
app = FastAPI(title="LensWord Sentiment Analysis API")

# Define the LSTM model architecture (2-layer Bidirectional LSTM)
class LensWordLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers,
                 num_classes, dropout):
        super(LensWordLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        return output

# Load the vocabulary
with open('../data/word2idx.pkl', 'rb') as f:
    word2idx = pickle.load(f)

# Load all 3 ensemble models
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
    m.load_state_dict(torch.load(f'../models/lensword_model_seed{seed}.pt',
                                  map_location=torch.device('cpu')))
    m.eval()
    ensemble_models.append(m)

print(f"Loaded {len(ensemble_models)} ensemble models successfully!")

# Request format
class ReviewRequest(BaseModel):
    text: str

# Text preprocessing functions
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def text_to_sequence(text):
    words = text.split()
    sequence = [word2idx.get(word, word2idx['<UNK>']) for word in words]
    return sequence

def pad_sequence(sequence, max_length):
    if len(sequence) < max_length:
        sequence = sequence + [0] * (max_length - len(sequence))
    else:
        sequence = sequence[:max_length]
    return sequence

# Health check endpoint
@app.get("/")
def home():
    return {
        "message": "LensWord Sentiment Analysis API is running!",
        "model_type": "Ensemble (3 Bidirectional LSTM models)",
        "seeds": SEEDS,
        "classes": ["Negative", "Neutral", "Positive"]
    }

# Prediction endpoint
@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
    # Preprocess
    cleaned = clean_text(request.text)
    sequence = text_to_sequence(cleaned)
    padded = pad_sequence(sequence, MAX_SEQ_LENGTH)
    input_tensor = torch.tensor([padded], dtype=torch.long)

    # Run all 3 models and average probabilities
    all_probs = []
    with torch.no_grad():
        for m in ensemble_models:
            output = m(input_tensor)
            probs = torch.softmax(output, dim=1)
            all_probs.append(probs)

    avg_probs = torch.stack(all_probs).mean(dim=0)
    predicted_class = avg_probs.argmax(dim=1).item()
    confidence = avg_probs[0][predicted_class].item()

    labels = ['Negative', 'Neutral', 'Positive']
    sentiment = labels[predicted_class]

    # Return all probabilities for transparency
    return {
        "text": request.text,
        "sentiment": sentiment,
        "confidence": round(confidence * 100, 2),
        "probabilities": {
            "Negative": round(avg_probs[0][0].item() * 100, 2),
            "Neutral": round(avg_probs[0][1].item() * 100, 2),
            "Positive": round(avg_probs[0][2].item() * 100, 2)
        },
        "model_type": "ensemble_3_models"
    }