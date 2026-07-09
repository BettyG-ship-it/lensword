# LensWord - FastAPI Sentiment Prediction API
# LSTM + RAG + Groq LLM + LangGraph + SQLite + WebSocket

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
from lime.lime_text import LimeTextExplainer

load_dotenv()

sys.path.append(os.path.abspath('..'))
from src.config import *
from model import LensWordLSTM
from database import (init_db, save_prediction, save_message,
                      generate_ticket_id, get_all_tickets, get_ticket,
                      get_stats, update_status, save_satisfaction,
                      get_alerts, check_drift)
from conversation import run_conversation, continue_conversation

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ── Global state ─────────────────────────────────────────────
app_state = {}

# ── Conversation step tracker ─────────────────────────────────
# Keeps track of each ticket's current conversation step
conversation_steps = {}

# ── WebSocket manager ─────────────────────────────────────────
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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    init_db()

    vocab_path = os.path.join(base_dir, '..', 'data', 'word2idx.pkl')
    with open(vocab_path, 'rb') as f:
        app_state['word2idx'] = pickle.load(f)
    print("Vocabulary loaded!")

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

    kb_path = os.path.join(base_dir, 'knowledge_base.csv')
    kb_df   = pd.read_csv(kb_path)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    app_state['embedder'] = embedder

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

    if os.environ.get("GROQ_API_KEY"):
        print("Groq LLM ready!")
    else:
        print("Warning: GROQ_API_KEY not found")

    app_state['ready'] = True
    yield

    app_state.clear()
    conversation_steps.clear()
    print("API shutdown complete.")

app = FastAPI(
    title="LensWord API — LSTM + RAG + Groq + LangGraph",
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

# ── Groq: Is this a product review? ─────────────────────────
def is_product_review_groq(text: str) -> bool:
    """Use Groq to detect if text is a product/service review"""
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=5,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You decide if text is a product or service review. Reply only YES or NO."
                },
                {
                    "role": "user",
                    "content": f"Is this a product or service review?\n\n'{text}'"
                }
            ]
        )
        answer = completion.choices[0].message.content.strip().upper()
        return "YES" in answer
    except Exception:
        # Fallback to vocabulary check if Groq fails
        words = text.lower().split()
        if len(words) < 4:
            return False
        review_indicators = [
            'product', 'item', 'order', 'bought', 'purchased', 'shipping',
            'delivery', 'quality', 'arrived', 'received', 'return', 'refund',
            'broken', 'works', 'love', 'hate', 'great', 'awful', 'terrible',
            'amazing', 'good', 'bad', 'excellent', 'poor', 'disappointed',
            'satisfied', 'recommend', 'waste', 'money', 'price', 'worth',
            'service', 'customer', 'seller', 'package', 'damaged', 'missing',
            'wrong', 'perfect', 'happy', 'unhappy', 'experience', 'use',
            'problem', 'issue', 'defective', 'stopped', 'working'
        ]
        return any(w in review_indicators for w in words)

# ── Groq: Does customer want a human agent? ──────────────────
def should_escalate_groq(text: str) -> bool:
    """Use Groq to detect escalation intent — no fixed word list"""
    # Must be at least 3 words to be an escalation request
    if len(text.strip().split()) < 3:
        return False

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=5,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You detect ONLY if a customer explicitly wants to speak to "
                        "a human agent or live support person. "
                        "Do NOT flag complaints or negative reviews — only escalation requests. "
                        "Reply only YES or NO."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Does this message explicitly request a human agent or live support person?\n\n'{text}'"
                    )
                }
            ]
        )
        answer = completion.choices[0].message.content.strip().upper()
        return "YES" in answer
    except Exception:
        # Fallback keywords if Groq fails
        triggers = ['speak to a human', 'talk to a person', 'real person',
                    'human agent', 'live agent', 'speak to someone',
                    'talk to someone', 'supervisor', 'manager']
        return any(t in text.lower() for t in triggers)

# ── Groq: Has sentiment shifted mid-conversation? ────────────
def detect_sentiment_shift(text: str, original_sentiment: str) -> Optional[str]:
    """Detect if customer sentiment has shifted during conversation"""
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=10,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You detect customer sentiment in one word. "
                        "Reply only: NEGATIVE, NEUTRAL, or POSITIVE."
                    )
                },
                {
                    "role": "user",
                    "content": f"What is the sentiment of this message?\n\n'{text}'"
                }
            ]
        )
        new_sentiment = completion.choices[0].message.content.strip().upper()
        if "NEGATIVE" in new_sentiment:
            new_sentiment = "Negative"
        elif "NEUTRAL" in new_sentiment:
            new_sentiment = "Neutral"
        else:
            new_sentiment = "Positive"

        # Return new sentiment only if it shifted significantly
        if new_sentiment != original_sentiment:
            return new_sentiment
        return None
    except Exception:
        return None

# ── RAG context retrieval ────────────────────────────────────
def get_rag_context(review_text: str, sentiment: str) -> str:
    fallback = {
        'Negative': 'We are sorry to hear about your experience. A representative will contact you within 24 hours.',
        'Neutral':  'Thank you for your feedback. We would love to know how we can improve.',
        'Positive': 'Thank you for your wonderful feedback! We are thrilled you enjoyed your purchase.'
    }

    if sentiment == 'Positive':
        return fallback['Positive']

    try:
        embedder        = app_state['embedder']
        collection      = app_state['collection']
        query_embedding = embedder.encode([review_text]).tolist()
        results         = collection.query(query_embeddings=query_embedding, n_results=1)
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

# ── Groq LLM response generation ────────────────────────────
def generate_llm_response(review_text: str, sentiment: str, rag_context: str) -> str:
    try:
        if not os.environ.get("GROQ_API_KEY"):
            return rag_context

        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=120,
            temperature=0.7,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a warm empathetic customer service agent. "
                        "Read the customer review carefully and respond appropriately. "
                        "If the review describes a problem — apologize and offer help. "
                        "If positive — respond with genuine appreciation. "
                        "Use the provided policy as your guide. Never invent information. "
                        "Keep response to 2-3 sentences. Sound human and genuine."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Customer review: {review_text}\n"
                        f"Policy/context: {rag_context}\n\n"
                        f"Write a personalized empathetic response."
                    )
                }
            ]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq LLM error: {e}")
        return rag_context

# ── Request models ───────────────────────────────────────────
class ReviewRequest(BaseModel):
    text:               str = Field(..., min_length=1, max_length=2000)
    session_id:         Optional[str] = None
    override_sentiment: Optional[str] = None

class ChatRequest(BaseModel):
    ticket_id:          str
    message:            str
    current_step:       str
    review_text:        str
    original_sentiment: str

class EscalateRequest(BaseModel):
    ticket_id: str
    message:   str

class MessageRequest(BaseModel):
    ticket_id: str
    sender:    str
    message:   str

class StatusRequest(BaseModel):
    ticket_id: str
    status:    str

class SatisfactionRequest(BaseModel):
    ticket_id: str
    rating:    int = Field(..., ge=1, le=5)

# ── LIME Explainability ─────────────────────────────────────
def get_explanation(text: str, num_features: int = 6) -> list:
    """Use LIME to find which words drove the prediction"""
    try:
        word2idx = app_state['word2idx']
        model    = app_state['model']

        # First get the predicted class
        cleaned  = clean_text(text)
        sequence = text_to_sequence(cleaned, word2idx)
        padded   = pad_sequence(sequence, MAX_SEQ_LENGTH)
        tensor   = torch.tensor([padded], dtype=torch.long)
        with torch.no_grad():
            output = model(tensor)
            probs  = torch.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()

        def predict_proba(texts):
            results = []
            for t in texts:
                try:
                    c  = clean_text(t)
                    sq = text_to_sequence(c, word2idx)
                    pd = pad_sequence(sq, MAX_SEQ_LENGTH)
                    tn = torch.tensor([pd], dtype=torch.long)
                    with torch.no_grad():
                        out   = model(tn)
                        probs = torch.softmax(out, dim=1)
                    results.append(probs[0].detach().numpy())
                except Exception:
                    results.append([0.33, 0.33, 0.34])
            import numpy as np
            return np.array(results)

        explainer   = LimeTextExplainer(
            class_names=['Negative', 'Neutral', 'Positive'],
            bow=True
        )
        explanation = explainer.explain_instance(
            text,
            predict_proba,
            num_features=num_features,
            num_samples=200,
            labels=[predicted_class]
        )

        # Get word importances for predicted class
        word_importances = explanation.as_list(label=predicted_class)

        if not word_importances:
            print("LIME returned no word importances")
            return []

        # Stop words to filter out — not meaningful for sentiment
        stop_words = {
            'the','a','an','this','that','is','was','are','were',
            'it','its','i','my','me','we','our','you','your',
            'to','of','in','on','at','for','with','and','or',
            'but','not','so','as','by','be','do','did','has',
            'had','have','will','would','could','should','may',
            'might','shall','after','before','two','one','three',
            'days','day','week','weeks','month','months','year','years',
            'just','very','really','quite','also','even','still','already'
        }

        labels_map = ['Negative', 'Neutral', 'Positive']
        predicted_label = labels_map[predicted_class]

        # Format for frontend
        result = []
        for word, importance in word_importances:
            # Skip stop words and near-zero importances
            if word.lower() in stop_words or word.isdigit():
                continue
            if abs(importance) < 0.005:
                continue
            result.append({
                "word":       word,
                "importance": round(abs(importance) * 100, 1),
                "drives":     predicted_label,
                "direction":  "toward" if importance > 0 else "against"
            })

        result.sort(key=lambda x: x["importance"], reverse=True)
        return result[:6]

    except Exception as e:
        print(f"LIME explanation error: {e}")
        import traceback
        traceback.print_exc()
        return []

class ExplainRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)

@app.post("/explain")
def explain_prediction(request: ExplainRequest):
    """Return word importance scores for a prediction"""
    explanation = get_explanation(request.text)
    return {
        "text":        request.text,
        "explanation": explanation
    }

# ── Health check ─────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "message": "LensWord API — LSTM + RAG + Groq + LangGraph",
        "status":  "ready" if app_state.get('ready') else "initializing",
        "llm":     "Groq llama-3.1-8b-instant" if os.environ.get("GROQ_API_KEY") else "RAG fallback"
    }

# ── Main prediction endpoint ─────────────────────────────────
@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
    word2idx = app_state['word2idx']
    model    = app_state['model']

    cleaned = clean_text(request.text)
    if not cleaned.strip():
        return {
            "text": request.text, "sentiment": None, "confidence": 0.0,
            "probabilities": {"Negative": 0.0, "Neutral": 0.0, "Positive": 0.0},
            "action": "cannot_score", "priority": "UNKNOWN", "ticket_id": None,
            "suggested_response": "Thank you for reaching out! This chat helps with product and service feedback."
        }

    # Check escalation FIRST — before out-of-domain check
    if should_escalate_groq(cleaned):
        ticket_id = generate_ticket_id()
        save_prediction(
            ticket_id=ticket_id,
            review_text=request.text,
            sentiment='Negative',
            confidence=100.0,
            priority='HIGH',
            action='escalate',
            suggested_response='Customer requested human agent directly.',
            session_id=request.session_id
        )
        save_message(ticket_id, 'customer', request.text)
        update_status(ticket_id, 'escalated')
        return {
            "text": request.text, "sentiment": "Negative", "confidence": 100.0,
            "probabilities": {"Negative": 100.0, "Neutral": 0.0, "Positive": 0.0},
            "action": "escalate_direct", "priority": "HIGH",
            "ticket_id": ticket_id,
            "lang_messages": [{
                "sender": "bot",
                "message": "I completely understand. Let me connect you with one of our senior support agents right away. They will have full context of our conversation and will be with you very shortly.",
                "timestamp": ""
            }],
            "current_step": "escalated",
            "suggested_response": "Customer requested human agent directly."
        }

    # Groq detects if it is a product review
    if not is_product_review_groq(cleaned):
        return {
            "text": request.text, "sentiment": None, "confidence": 0.0,
            "probabilities": {"Negative": 0.0, "Neutral": 0.0, "Positive": 0.0},
            "action": "cannot_score", "priority": "UNKNOWN", "ticket_id": None,
            "suggested_response": "Thank you for reaching out! This chat helps with product and service feedback."
        }

    # LSTM prediction
    sequence     = text_to_sequence(cleaned, word2idx)
    padded       = pad_sequence(sequence, MAX_SEQ_LENGTH)
    input_tensor = torch.tensor([padded], dtype=torch.long)

    with torch.no_grad():
        output = model(input_tensor)
        probs  = torch.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probs[0][predicted_class].item()

    labels    = ['Negative', 'Neutral', 'Positive']
    sentiment = labels[predicted_class]

    actions = {
        'Positive': {'action': 'none',      'priority': 'LOW'},
        'Neutral':  {'action': 'follow_up', 'priority': 'MEDIUM'},
        'Negative': {'action': 'escalate',  'priority': 'HIGH'}
    }

    # Compute final sentiment BEFORE RAG so everything uses the correct sentiment
    final_sentiment = request.override_sentiment if request.override_sentiment else sentiment
    final_priority  = actions[final_sentiment]['priority']
    final_action    = actions[final_sentiment]['action']

    # RAG + Groq LLM response — uses final_sentiment so override is respected
    rag_context        = get_rag_context(request.text, final_sentiment)
    suggested_response = generate_llm_response(request.text, final_sentiment, rag_context)

    ticket_id = generate_ticket_id()
    save_prediction(
        ticket_id=ticket_id,
        review_text=request.text,
        sentiment=final_sentiment,
        confidence=round(confidence * 100, 2),
        priority=final_priority,
        action=final_action,
        suggested_response=suggested_response,
        session_id=request.session_id
    )
    save_message(ticket_id, 'customer', request.text)

    # Start LangGraph conversation using final sentiment
    lang_result = run_conversation(
        ticket_id=ticket_id,
        review_text=request.text,
        sentiment=final_sentiment,
        confidence=round(confidence * 100, 2)
    )

    # Store current step
    conversation_steps[ticket_id] = {
        "step":      lang_result["step"],
        "sentiment": final_sentiment,
        "review":    request.text
    }

    try:
        check_drift()
    except Exception:
        pass

    # Clean up old steps — keep only last 1000 tickets in memory
    if len(conversation_steps) > 1000:
        oldest_keys = list(conversation_steps.keys())[:100]
        for k in oldest_keys:
            conversation_steps.pop(k, None)

    return {
        "text":               request.text,
        "sentiment":          final_sentiment,
        "confidence":         round(confidence * 100, 2),
        "probabilities": {
            "Negative": round(probs[0][0].item() * 100, 2),
            "Neutral":  round(probs[0][1].item() * 100, 2),
            "Positive": round(probs[0][2].item() * 100, 2)
        },
        "action":             final_action,
        "priority":           final_priority,
        "suggested_response": suggested_response,
        "ticket_id":          ticket_id,
        "lang_messages":      lang_result["messages"],
        "current_step":       lang_result["step"]
    }

# ── Chat continue endpoint ───────────────────────────────────
@app.post("/chat/continue")
async def chat_continue(request: ChatRequest):
    """Continue LangGraph conversation — called when customer replies"""

    # Check if customer wants to escalate using Groq
    wants_human = should_escalate_groq(request.message)
    if wants_human:
        update_status(request.ticket_id, 'escalated')
        save_message(request.ticket_id, 'customer', request.message)
        await manager.broadcast(request.ticket_id, {
            "type":      "escalation",
            "ticket_id": request.ticket_id,
            "message":   "Customer requested a human agent"
        })
        return {
            "messages": [{
                "sender":  "bot",
                "message": "I completely understand. Let me connect you with one of our senior support agents right away. They will be with you very shortly and will have full context of our conversation.",
                "timestamp": datetime.now().isoformat()
            }],
            "step":        "escalated",
            "needs_agent": True,
            "resolved":    False,
            "escalated":   True
        }

    # Check for sentiment shift
    new_sentiment = detect_sentiment_shift(request.message, request.original_sentiment)

    # Save customer message
    save_message(request.ticket_id, 'customer', request.message)

    # Continue LangGraph conversation
    result = continue_conversation(
        ticket_id=request.ticket_id,
        review_text=request.review_text,
        sentiment=new_sentiment or request.original_sentiment,
        current_step=request.current_step,
        customer_message=request.message
    )

    # Save bot messages
    for msg in result["messages"]:
        save_message(request.ticket_id, 'bot', msg["message"])

    # Update step tracker
    conversation_steps[request.ticket_id] = {
        "step":      result["step"],
        "sentiment": new_sentiment or request.original_sentiment,
        "review":    request.review_text
    }

    # Update ticket status if resolved
    if result["resolved"]:
        update_status(request.ticket_id, 'resolved')

    return {
        "messages":      result["messages"],
        "step":          result["step"],
        "needs_agent":   result["needs_agent"],
        "resolved":      result["resolved"],
        "escalated":     False,
        "sentiment_shift": new_sentiment
    }

# ── Escalate endpoint ────────────────────────────────────────
@app.post("/chat/escalate")
async def escalate_chat(request: EscalateRequest):
    update_status(request.ticket_id, 'escalated')
    save_message(request.ticket_id, 'customer', request.message)
    await manager.broadcast(request.ticket_id, {
        "type":      "escalation",
        "ticket_id": request.ticket_id,
        "message":   "Customer requested a human agent"
    })
    return {"status": "escalated", "ticket_id": request.ticket_id}

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
    return {"status": "saved", "rating": request.rating}

# ── Status endpoint ──────────────────────────────────────────
@app.post("/ticket/status")
def update_ticket_status(request: StatusRequest):
    update_status(request.ticket_id, request.status)
    return {"status": "updated"}

# ── Admin endpoints ──────────────────────────────────────────
@app.get("/admin/stats")
def admin_stats():
    stats = get_stats()
    # Remove None keys that break the dashboard
    stats["total_counts"] = {k: v for k, v in stats["total_counts"].items() if k is not None}
    stats["today_counts"] = {k: v for k, v in stats["today_counts"].items() if k is not None}
    stats["priority_counts"] = {k: v for k, v in stats["priority_counts"].items() if k is not None}
    return stats

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
