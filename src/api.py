# LensWord - FastAPI Sentiment Prediction API with RAG
# Uses Bidirectional LSTM for sentiment + RAG for response suggestions

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
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

# ── Global state ─────────────────────────────────────────────
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load vocabulary
    vocab_path = os.path.join(base_dir, '..', 'data', 'word2idx.pkl')
    with open(vocab_path, 'rb') as f:
        app_state['word2idx'] = pickle.load(f)
    print("Vocabulary loaded!")

    # Load LSTM model
    model = LensWordLSTM(
        vocab_size=MAX_VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT
    )
    model_path = os.path.join(base_dir, '..', 'models', 'lensword_model.pt')
    model.load_state_dict(torch.load(
        model_path,
        map_location=torch.device('cpu'),
        weights_only=True
    ))
    model.eval()
    app_state['model'] = model
    print("LSTM model loaded!")

    # Setup RAG
    kb_path = os.path.join(base_dir, 'knowledge_base.csv')
    kb_df = pd.read_csv(kb_path)

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    app_state['embedder'] = embedder

    # Force recreate collection every startup
    chroma_client = chromadb.Client()
    try:
        chroma_client.delete_collection(name="lensword_kb")
    except Exception:
        pass

    collection = chroma_client.create_collection(name="lensword_kb")
    documents  = kb_df['complaint_example'].tolist()
    responses  = kb_df['response'].tolist()
    ids        = [f"doc_{i}" for i in range(len(documents))]
    embeddings = embedder.encode(documents).tolist()
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"response": r} for r in responses]
    )

    app_state['collection'] = collection
    print(f"RAG knowledge base loaded with {len(documents)} entries!")

    app_state['ready'] = True
    yield

    # ── Shutdown ─────────────────────────────────────────────
    app_state.clear()
    print("API shutdown complete.")

# ── Create FastAPI app ───────────────────────────────────────
app = FastAPI(
    title="LensWord Sentiment Analysis API with RAG",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Text preprocessing ───────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def text_to_sequence(text: str, word2idx: dict) -> list:
    words = text.split()
    return [word2idx.get(word, word2idx['<UNK>']) for word in words]

def pad_sequence(sequence: list, max_length: int) -> list:
    if len(sequence) < max_length:
        sequence = sequence + [0] * (max_length - len(sequence))
    else:
        sequence = sequence[:max_length]
    return sequence

# ── RAG response retrieval ───────────────────────────────────
def get_rag_response(review_text: str, sentiment: str) -> str:
    fallback = {
        'Negative': 'We are sorry to hear about your experience. A customer service representative will contact you within 24 hours to resolve this for you.',
        'Neutral':  'Thank you for your feedback. We would love to know how we can improve your experience further.',
        'Positive': 'Thank you so much for your wonderful feedback! We are thrilled you are enjoying your purchase. We look forward to serving you again soon.'
    }

    if sentiment == 'Positive':
        return fallback['Positive']

    try:
        embedder   = app_state['embedder']
        collection = app_state['collection']
        query_embedding = embedder.encode([review_text]).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=1
        )
        if (results
                and results.get('documents')
                and len(results['documents']) > 0
                and len(results['documents'][0]) > 0
                and results.get('metadatas')
                and len(results['metadatas']) > 0
                and len(results['metadatas'][0]) > 0):
            response = results['metadatas'][0][0].get('response', '')
            if response:
                return response
    except Exception as e:
        print(f"RAG query failed: {e}")

    return fallback.get(sentiment, 'Thank you for your feedback.')

# ── Out-of-domain detection ───────────────────────────────────
def is_product_review(text: str) -> bool:
    """
    Basic heuristic to detect if input is likely a product/service review.
    Returns False for very short text or text with no review-like vocabulary.
    """
    words = text.lower().split()

    # Too short to be meaningful
    if len(words) < 4:
        return False

    # Check for at least some review-like vocabulary
    review_indicators = [
        'product', 'item', 'order', 'bought', 'purchased', 'shipping', 'delivery',
        'quality', 'arrived', 'received', 'return', 'refund', 'broken', 'works',
        'love', 'hate', 'great', 'awful', 'terrible', 'amazing', 'good', 'bad',
        'excellent', 'poor', 'disappointed', 'satisfied', 'recommend', 'waste',
        'money', 'price', 'worth', 'service', 'customer', 'seller', 'package',
        'box', 'damaged', 'missing', 'wrong', 'perfect', 'happy', 'unhappy',
        'experience', 'buy', 'buying', 'use', 'using', 'like', 'dislike',
        'nice', 'cheap', 'expensive', 'fast', 'slow', 'easy', 'difficult',
        'problem', 'issue', 'defective', 'broken', 'stopped', 'working'
    ]

    # If at least one review indicator is present — likely a review
    for word in words:
        if word in review_indicators:
            return True

    return False

# ── Request model ────────────────────────────────────────────
class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000,
                      description="Review text to analyze (1-2000 characters)")

# ── Health check endpoint ────────────────────────────────────
@app.get("/")
def home():
    return {
        "message": "LensWord Sentiment Analysis API with RAG is running!",
        "model":   "Bidirectional LSTM + RAG Response System",
        "classes": ["Negative", "Neutral", "Positive"],
        "status":  "ready" if app_state.get('ready') else "initializing"
    }

# ── Prediction endpoint ──────────────────────────────────────
@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
    word2idx = app_state['word2idx']
    model    = app_state['model']

    # F4 fix: empty input guard
    cleaned = clean_text(request.text)
    if not cleaned.strip():
        return {
            "text":       request.text,
            "sentiment":  None,
            "confidence": 0.0,
            "probabilities": {"Negative": 0.0, "Neutral": 0.0, "Positive": 0.0},
            "action":    "cannot_score",
            "priority":  "UNKNOWN",
            "suggested_response": "Thank you for reaching out! This chat is designed to help with product and service feedback. Could you tell us about your experience with our product?"
        }

    # Out-of-domain guard — politely redirect non-review input
    if not is_product_review(cleaned):
        return {
            "text":       request.text,
            "sentiment":  None,
            "confidence": 0.0,
            "probabilities": {"Negative": 0.0, "Neutral": 0.0, "Positive": 0.0},
            "action":    "cannot_score",
            "priority":  "UNKNOWN",
            "suggested_response": "Thank you for reaching out! This chat is designed to help with product and service feedback. Could you tell us about your experience with our product?"
        }

    # Preprocess
    sequence     = text_to_sequence(cleaned, word2idx)
    padded       = pad_sequence(sequence, MAX_SEQ_LENGTH)
    input_tensor = torch.tensor([padded], dtype=torch.long)

    # LSTM prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs  = torch.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probs[0][predicted_class].item()

    labels    = ['Negative', 'Neutral', 'Positive']
    sentiment = labels[predicted_class]

    # RAG response
    suggested_response = get_rag_response(request.text, sentiment)

    # Action and priority
    actions = {
        'Positive': {'action': 'none',      'priority': 'LOW'},
        'Neutral':  {'action': 'follow_up', 'priority': 'MEDIUM'},
        'Negative': {'action': 'escalate',  'priority': 'HIGH'}
    }

    return {
        "text":       request.text,
        "sentiment":  sentiment,
        "confidence": round(confidence * 100, 2),
        "probabilities": {
            "Negative": round(probs[0][0].item() * 100, 2),
            "Neutral":  round(probs[0][1].item() * 100, 2),
            "Positive": round(probs[0][2].item() * 100, 2)
        },
        "action":             actions[sentiment]['action'],
        "priority":           actions[sentiment]['priority'],
        "suggested_response": suggested_response
    }
