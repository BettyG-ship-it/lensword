# LensWord - FastAPI Sentiment Prediction API with RAG
# Uses Bidirectional LSTM for sentiment + RAG for response suggestions

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import pickle
import re
import os
import sys
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# Add project root to path
sys.path.append(os.path.abspath('..'))
from src.config import *

# Create FastAPI app
app = FastAPI(title="LensWord Sentiment Analysis API with RAG")

# Allow requests from index.html
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── LSTM Model Architecture ──────────────────────────────────
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

# ── Load vocabulary ──────────────────────────────────────────
with open('../data/word2idx.pkl', 'rb') as f:
    word2idx = pickle.load(f)

# ── Load LSTM model ──────────────────────────────────────────
model = LensWordLSTM(
    vocab_size=MAX_VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
)
model.load_state_dict(torch.load(
    '../models/lensword_model.pt',
    map_location=torch.device('cpu')
))
model.eval()
print("LSTM model loaded successfully!")

# ── Setup RAG ────────────────────────────────────────────────
print("Setting up RAG knowledge base...")

# Load knowledge base
kb_path = os.path.join(os.path.dirname(__file__), 'knowledge_base.csv')
kb_df = pd.read_csv(kb_path)

# Load sentence transformer for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Setup ChromaDB
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="lensword_kb")

# Add knowledge base to ChromaDB
documents = kb_df['complaint_example'].tolist()
responses = kb_df['response'].tolist()
ids = [f"doc_{i}" for i in range(len(documents))]
embeddings = embedder.encode(documents).tolist()

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=ids,
    metadatas=[{"response": r} for r in responses]
)
print(f"RAG knowledge base loaded with {len(documents)} entries!")

# ── Text preprocessing functions ─────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def text_to_sequence(text):
    words = text.split()
    return [word2idx.get(word, word2idx['<UNK>']) for word in words]

def pad_sequence(sequence, max_length):
    if len(sequence) < max_length:
        sequence = sequence + [0] * (max_length - len(sequence))
    else:
        sequence = sequence[:max_length]
    return sequence

# ── RAG response retrieval ───────────────────────────────────
def get_rag_response(review_text, sentiment):
    if sentiment == 'Positive':
        return "Thank you so much for your wonderful feedback! We are thrilled you are enjoying your purchase. We look forward to serving you again soon."

    query_embedding = embedder.encode([review_text]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=1
    )

    if results and results['metadatas']:
        return results['metadatas'][0][0]['response']

    fallback = {
        'Negative': 'We are sorry to hear about your experience. A customer service representative will contact you within 24 hours.',
        'Neutral': 'Thank you for your feedback. We would love to know how we can improve your experience further.'
    }
    return fallback[sentiment]

# ── Request format ───────────────────────────────────────────
class ReviewRequest(BaseModel):
    text: str

# ── Health check endpoint ────────────────────────────────────
@app.get("/")
def home():
    return {
        "message": "LensWord Sentiment Analysis API with RAG is running!",
        "model": "Bidirectional LSTM + RAG Response System",
        "classes": ["Negative", "Neutral", "Positive"]
    }

# ── Prediction endpoint ──────────────────────────────────────
@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
    # Step 1 - Preprocess text
    cleaned = clean_text(request.text)
    sequence = text_to_sequence(cleaned)
    padded = pad_sequence(sequence, MAX_SEQ_LENGTH)
    input_tensor = torch.tensor([padded], dtype=torch.long)

    # Step 2 - LSTM prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probs[0][predicted_class].item()

    labels = ['Negative', 'Neutral', 'Positive']
    sentiment = labels[predicted_class]

    # Step 3 - RAG response retrieval
    suggested_response = get_rag_response(request.text, sentiment)

    # Step 4 - Action based on sentiment
    actions = {
        'Positive': {'action': 'none', 'priority': 'LOW'},
        'Neutral': {'action': 'follow_up', 'priority': 'MEDIUM'},
        'Negative': {'action': 'escalate', 'priority': 'HIGH'}
    }

    return {
        "text": request.text,
        "sentiment": sentiment,
        "confidence": round(confidence * 100, 2),
        "probabilities": {
            "Negative": round(probs[0][0].item() * 100, 2),
            "Neutral": round(probs[0][1].item() * 100, 2),
            "Positive": round(probs[0][2].item() * 100, 2)
        },
        "action": actions[sentiment]['action'],
        "priority": actions[sentiment]['priority'],
        "suggested_response": suggested_response
    }
