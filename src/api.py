# LensWord - FastAPI Sentiment Prediction API with RAG + Groq LLM
# Uses Bidirectional LSTM for sentiment + RAG for context + Groq for personalized responses

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import torch
import pickle
import re
import os
import sys
import json
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from typing import Optional
from datetime import datetime
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# Add project root to path
sys.path.append(os.path.abspath('..'))
from src.config import *
from model import LensWordLSTM
from database import (init_db, save_prediction, save_message,
                      generate_ticket_id, get_all_tickets, get_ticket,
                      get_stats, update_status, save_satisfaction,
                      get_alerts, check_drift)

# Initialize Groq client
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ── Global state ─────────────────────────────────────────────
app_state = {}

# ── WebSocket connection manager ─────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: dict = {}

    async def connect(self, ticket_id: str, websocket: WebSocket):
        await websocket.accept()
        if ticket_id not in self.active:
            self.active[ticket_id] = []
        self.active[ticket_id].append(websocket)

    def disconnect(self, ticket_id: str, websocket: WebSocket):
        if ticket_id in self.active:
            self.active[ticket_id].remove(websocket)

    async def broadcast(self, ticket_id: str, message: dict):
        if ticket_id in self.active:
            for ws in self.active[ticket_id]:
                try:
                    await ws.send_text(json.dumps(message))
                except Exception:
                    pass

manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Initialize SQLite database
    init_db()

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

    # Force recreate ChromaDB collection
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

    # Check Groq is working
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        print("Groq LLM ready!")
    else:
        print("Warning: GROQ_API_KEY not found — falling back to RAG templates")

    app_state['ready'] = True
    yield

    # ── Shutdown ─────────────────────────────────────────────
    app_state.clear()
    print("API shutdown complete.")

# ── Create FastAPI app ───────────────────────────────────────
app = FastAPI(
    title="LensWord Sentiment Analysis API with RAG + LLM",
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

# ── Out-of-domain detection ───────────────────────────────────
def is_product_review(text: str) -> bool:
    words = text.lower().split()
    if len(words) < 4:
        return False
    review_indicators = [
        'product', 'item', 'order', 'bought', 'purchased', 'shipping',
        'delivery', 'quality', 'arrived', 'received', 'return', 'refund',
        'broken', 'works', 'love', 'hate', 'great', 'awful', 'terrible',
        'amazing', 'good', 'bad', 'excellent', 'poor', 'disappointed',
        'satisfied', 'recommend', 'waste', 'money', 'price', 'worth',
        'service', 'customer', 'seller', 'package', 'box', 'damaged',
        'missing', 'wrong', 'perfect', 'happy', 'unhappy', 'experience',
        'buy', 'buying', 'use', 'using', 'like', 'dislike', 'nice',
        'cheap', 'expensive', 'fast', 'slow', 'easy', 'difficult',
        'problem', 'issue', 'defective', 'stopped', 'working'
    ]
    for word in words:
        if word in review_indicators:
            return True
    return False

# ── RAG retrieval ────────────────────────────────────────────
def get_rag_context(review_text: str, sentiment: str) -> str:
    fallback = {
        'Negative': 'We are sorry to hear about your experience. A customer service representative will contact you within 24 hours.',
        'Neutral':  'Thank you for your feedback. We would love to know how we can improve your experience further.',
        'Positive': 'Thank you so much for your wonderful feedback! We are thrilled you are enjoying your purchase.'
    }

    if sentiment == 'Positive':
        return fallback['Positive']

    try:
        embedder   = app_state['embedder']
        collection = app_state['collection']
        query_embedding = embedder.encode([review_text]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=1)
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

# ── Groq LLM response generation ─────────────────────────────
def generate_llm_response(review_text: str, sentiment: str, rag_context: str) -> str:
    try:
        groq_key = os.environ.get("GROQ_API_KEY")
        if not groq_key:
            return rag_context
        
        print(f"Calling Groq LLM for: {review_text[:50]}...")

        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=120,
            temperature=0.7,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a warm, empathetic customer service agent for an e-commerce company. "
                        "Your job is to respond to customer reviews in a personal, caring way. "
                        "Always use the provided policy as your guide. "
                        "Never invent information not in the policy. "
                        "Keep your response to 2-3 sentences maximum. "
                        "Sound human and genuine, not robotic or corporate."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Customer review: {review_text}\n"
                        f"Sentiment detected: {sentiment}\n"
                        f"Relevant policy/context: {rag_context}\n\n"
                        f"Write a personalized, empathetic response to this customer. "
                        f"Use the policy above as your guide but make it feel personal to their specific situation."
                    )
                }
            ]
        )
        return completion.choices[0].message.content.strip()

    except Exception as e:
        print(f"Groq LLM error: {e} — falling back to RAG template")
        return rag_context  # Fallback to RAG if LLM fails

# ── Request models ───────────────────────────────────────────
class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None

class MessageRequest(BaseModel):
    ticket_id: str
    sender: str
    message: str

class StatusRequest(BaseModel):
    ticket_id: str
    status: str

class SatisfactionRequest(BaseModel):
    ticket_id: str
    rating: int = Field(..., ge=1, le=5)

# ── Health check ─────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "message": "LensWord API with RAG + Groq LLM is running!",
        "model":   "Bidirectional LSTM + RAG + Groq LLM",
        "classes": ["Negative", "Neutral", "Positive"],
        "llm":     "Groq llama3-8b-8192" if os.environ.get("GROQ_API_KEY") else "RAG fallback",
        "status":  "ready" if app_state.get('ready') else "initializing"
    }

# ── Prediction endpoint ──────────────────────────────────────
@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
    word2idx = app_state['word2idx']
    model    = app_state['model']

    # Empty input guard
    cleaned = clean_text(request.text)
    if not cleaned.strip():
        return {
            "text":       request.text,
            "sentiment":  None,
            "confidence": 0.0,
            "probabilities": {"Negative": 0.0, "Neutral": 0.0, "Positive": 0.0},
            "action":    "cannot_score",
            "priority":  "UNKNOWN",
            "suggested_response": "Thank you for reaching out! This chat is designed to help with product and service feedback.",
            "ticket_id": None
        }

    # Out-of-domain guard
    if not is_product_review(cleaned):
        return {
            "text":       request.text,
            "sentiment":  None,
            "confidence": 0.0,
            "probabilities": {"Negative": 0.0, "Neutral": 0.0, "Positive": 0.0},
            "action":    "cannot_score",
            "priority":  "UNKNOWN",
            "suggested_response": "Thank you for reaching out! This chat is designed to help with product and service feedback.",
            "ticket_id": None
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

    # Step 1 — RAG retrieves relevant context
    rag_context = get_rag_context(request.text, sentiment)

    # Step 2 — Groq LLM personalizes the response using RAG context
    suggested_response = generate_llm_response(request.text, sentiment, rag_context)

    # Action and priority
    actions = {
        'Positive': {'action': 'none',      'priority': 'LOW'},
        'Neutral':  {'action': 'follow_up', 'priority': 'MEDIUM'},
        'Negative': {'action': 'escalate',  'priority': 'HIGH'}
    }

    # Generate ticket and save to SQLite
    ticket_id = generate_ticket_id()
    save_prediction(
        ticket_id=ticket_id,
        review_text=request.text,
        sentiment=sentiment,
        confidence=round(confidence * 100, 2),
        priority=actions[sentiment]['priority'],
        action=actions[sentiment]['action'],
        suggested_response=suggested_response,
        session_id=request.session_id
    )
    save_message(ticket_id, 'customer', request.text)

    # Check for drift
    try:
        check_drift()
    except Exception:
        pass

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
        "suggested_response": suggested_response,
        "ticket_id":          ticket_id
    }

# ── Message endpoint ─────────────────────────────────────────
@app.post("/message")
async def save_chat_message(request: MessageRequest):
    save_message(request.ticket_id, request.sender, request.message)
    await manager.broadcast(request.ticket_id, {
        "sender":    request.sender,
        "message":   request.message,
        "timestamp": datetime.now().isoformat()
    })
    return {"status": "saved"}

# ── Satisfaction endpoint ────────────────────────────────────
@app.post("/satisfaction")
def rate_satisfaction(request: SatisfactionRequest):
    save_satisfaction(request.ticket_id, request.rating)
    return {"status": "saved", "ticket_id": request.ticket_id, "rating": request.rating}

# ── Status update endpoint ───────────────────────────────────
@app.post("/ticket/status")
def update_ticket_status(request: StatusRequest):
    update_status(request.ticket_id, request.status)
    return {"status": "updated", "ticket_id": request.ticket_id}

# ── Admin endpoints ──────────────────────────────────────────
@app.get("/admin/stats")
def admin_stats():
    return get_stats()

@app.get("/admin/tickets")
def admin_tickets(
    sentiment: str = None,
    priority:  str = None,
    status:    str = None,
    limit:     int = 50,
    offset:    int = 0
):
    return get_all_tickets(sentiment, priority, status, limit, offset)

@app.get("/admin/ticket/{ticket_id}")
def admin_ticket_detail(ticket_id: str):
    ticket = get_ticket(ticket_id)
    if not ticket:
        return {"error": "Ticket not found"}
    return ticket

@app.get("/admin/alerts")
def admin_alerts():
    return get_alerts()

# ── WebSocket endpoint ───────────────────────────────────────
@app.websocket("/ws/{ticket_id}")
async def websocket_endpoint(websocket: WebSocket, ticket_id: str):
    await manager.connect(ticket_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            msg  = json.loads(data)
            save_message(ticket_id, msg.get('sender', 'unknown'), msg.get('message', ''))
            await manager.broadcast(ticket_id, {
                "sender":    msg.get('sender'),
                "message":   msg.get('message'),
                "timestamp": datetime.now().isoformat()
            })
    except WebSocketDisconnect:
        manager.disconnect(ticket_id, websocket)