# LensWord -- LangGraph Conversation State Machine
# Manages customer support conversations with proper state tracking

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
from datetime import datetime
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ── Conversation State ────────────────────────────────────────
class ConversationState(TypedDict):
    ticket_id:      str
    review_text:    str
    sentiment:      str
    confidence:     float
    messages:       List[dict]
    step:           str
    resolved:       bool
    needs_agent:    bool
    customer_name:  Optional[str]
    order_number:   Optional[str]
    resolution:     Optional[str]

# ── Helper to add bot message ─────────────────────────────────
def bot_message(state: ConversationState, text: str) -> ConversationState:
    state["messages"].append({
        "sender":    "bot",
        "message":   text,
        "timestamp": datetime.now().isoformat()
    })
    return state

# ── Generate LLM response for a specific step ─────────────────
def generate_step_response(review_text: str, sentiment: str, step: str, context: str = "") -> str:
    prompts = {
        "greeting_negative": (
            f"Customer wrote: '{review_text}'\n"
            f"They are unhappy. Write a warm opening response acknowledging their frustration "
            f"and ask what went wrong. 2 sentences max."
        ),
        "collect_order": (
            f"Customer has a complaint about: '{review_text}'\n"
            f"Ask for their order number or email in a friendly way. 1-2 sentences."
        ),
        "offer_resolution": (
            f"Customer complained: '{review_text}'\n"
            f"Offer them options: replacement, full refund, or store credit. "
            f"Sound genuine and caring. 2 sentences max."
        ),
        "confirm_resolution": (
            f"Customer chose: {context}\n"
            f"For their complaint: '{review_text}'\n"
            f"Confirm their choice warmly and say a dedicated agent will contact them "
            f"within 24 hours. 2 sentences max."
        ),
        "closing_negative": (
            f"Close the conversation warmly. Thank them for their patience. "
            f"Promise to make it right. 2 sentences max."
        ),
        "greeting_neutral": (
            f"Customer wrote: '{review_text}'\n"
            f"They had an average experience. Ask warmly what could be improved. 2 sentences."
        ),
        "collect_feedback": (
            f"Customer said: {context}\n"
            f"Ask them to elaborate on what specifically could be better. 1-2 sentences."
        ),
        "closing_neutral": (
            f"Thank the customer for their honest feedback. Say their suggestions "
            f"will reach the product team. Sound genuine. 2 sentences."
        ),
        "greeting_positive": (
            f"Customer wrote: '{review_text}'\n"
            f"They are happy! Respond with genuine excitement and ask what they loved most. "
            f"2 sentences max."
        ),
        "ask_review": (
            f"Customer had a great experience. Ask them warmly if they would share "
            f"a review to help other customers. 1-2 sentences."
        ),
        "closing_positive": (
            f"Thank the customer enthusiastically. Say you look forward to serving them again. "
            f"Sound warm and genuine. 2 sentences max."
        ),
    }

    prompt = prompts.get(step, f"Respond warmly to: '{review_text}' in 2 sentences.")

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=100,
            temperature=0.7,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a warm, empathetic customer service agent. "
                        "Keep responses short, genuine and human. "
                        "Never use corporate jargon. Sound like a real person who cares."
                    )
                },
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"LangGraph LLM error: {e}")
        # Fallback responses
        fallbacks = {
            "greeting_negative": "We are really sorry to hear about your experience. Could you tell us more about what went wrong?",
            "collect_order": "Could you please share your order number or the email you used when purchasing?",
            "offer_resolution": "We would love to make this right for you. Would you prefer a replacement, a full refund, or store credit?",
            "closing_negative": "Thank you for your patience. We will make sure this gets resolved for you very soon.",
            "greeting_neutral": "Thank you for sharing your experience with us. What do you think we could do better?",
            "closing_neutral": "Thank you for your honest feedback. We will pass your suggestions to our team.",
            "greeting_positive": "That is wonderful to hear! We are so glad you had a great experience. What did you love most?",
            "closing_positive": "Thank you so much! We look forward to serving you again very soon.",
        }
        return fallbacks.get(step, "Thank you for reaching out. We are here to help!")

# ── NODE FUNCTIONS ────────────────────────────────────────────

def start_node(state: ConversationState) -> ConversationState:
    """Entry point — routes based on sentiment"""
    sentiment = state.get("sentiment", "Neutral")
    state["resolved"]    = False
    state["needs_agent"] = False

    if sentiment == "Negative":
        state["step"] = "negative_greeting"
    elif sentiment == "Neutral":
        state["step"] = "neutral_greeting"
    else:
        state["step"] = "positive_greeting"

    return state

def negative_greeting_node(state: ConversationState) -> ConversationState:
    response = generate_step_response(
        state["review_text"], "Negative", "greeting_negative"
    )
    state = bot_message(state, response)
    state["step"] = "collect_order"
    return state

def collect_order_node(state: ConversationState) -> ConversationState:
    response = generate_step_response(
        state["review_text"], "Negative", "collect_order"
    )
    state = bot_message(state, response)
    state["step"] = "offer_resolution"
    return state

def offer_resolution_node(state: ConversationState) -> ConversationState:
    response = generate_step_response(
        state["review_text"], "Negative", "offer_resolution"
    )
    state = bot_message(state, response)
    state["step"] = "confirm_resolution"
    return state

def confirm_resolution_node(state: ConversationState) -> ConversationState:
    resolution = state.get("resolution", "your preferred option")
    response = generate_step_response(
        state["review_text"], "Negative", "confirm_resolution", context=resolution
    )
    state = bot_message(state, response)
    state["step"] = "closing_negative"
    return state

def closing_negative_node(state: ConversationState) -> ConversationState:
    response = generate_step_response(
        state["review_text"], "Negative", "closing_negative"
    )
    state = bot_message(state, response)
    state["resolved"] = True
    state["step"]     = "resolved"
    return state

def neutral_greeting_node(state: ConversationState) -> ConversationState:
    response = generate_step_response(
        state["review_text"], "Neutral", "greeting_neutral"
    )
    state = bot_message(state, response)
    state["step"] = "collect_feedback"
    return state

def collect_feedback_node(state: ConversationState) -> ConversationState:
    response = generate_step_response(
        state["review_text"], "Neutral", "collect_feedback",
        context=state.get("resolution", "their feedback")
    )
    state = bot_message(state, response)
    state["step"] = "closing_neutral"
    return state

def closing_neutral_node(state: ConversationState) -> ConversationState:
    response = generate_step_response(
        state["review_text"], "Neutral", "closing_neutral"
    )
    state = bot_message(state, response)
    state["resolved"] = True
    state["step"]     = "resolved"
    return state

def positive_greeting_node(state: ConversationState) -> ConversationState:
    response = generate_step_response(
        state["review_text"], "Positive", "greeting_positive"
    )
    state = bot_message(state, response)
    state["step"] = "ask_review"
    return state

def ask_review_node(state: ConversationState) -> ConversationState:
    response = generate_step_response(
        state["review_text"], "Positive", "ask_review"
    )
    state = bot_message(state, response)
    state["step"] = "closing_positive"
    return state

def closing_positive_node(state: ConversationState) -> ConversationState:
    response = generate_step_response(
        state["review_text"], "Positive", "closing_positive"
    )
    state = bot_message(state, response)
    state["resolved"] = True
    state["step"]     = "resolved"
    return state

def escalate_node(state: ConversationState) -> ConversationState:
    state["needs_agent"] = True
    state["step"]        = "escalated"
    state = bot_message(state,
        "I completely understand your frustration. Let me connect you with one of our "
        "senior support agents right away who can give this the attention it deserves. "
        "Please hold on for just a moment."
    )
    return state

# ── Routing functions ─────────────────────────────────────────
def route_from_start(state: ConversationState) -> str:
    return state["step"]

def route_negative(state: ConversationState) -> str:
    step = state.get("step", "")
    if step == "collect_order":      return "collect_order"
    if step == "offer_resolution":   return "offer_resolution"
    if step == "confirm_resolution": return "confirm_resolution"
    if step == "closing_negative":   return "closing_negative"
    return END

def route_neutral(state: ConversationState) -> str:
    step = state.get("step", "")
    if step == "collect_feedback": return "collect_feedback"
    if step == "closing_neutral":  return "closing_neutral"
    return END

def route_positive(state: ConversationState) -> str:
    step = state.get("step", "")
    if step == "ask_review":        return "ask_review"
    if step == "closing_positive":  return "closing_positive"
    return END

# ── Build the graph ───────────────────────────────────────────
def build_graph():
    graph = StateGraph(ConversationState)

    # Add all nodes
    graph.add_node("start",              start_node)
    graph.add_node("negative_greeting",  negative_greeting_node)
    graph.add_node("collect_order",      collect_order_node)
    graph.add_node("offer_resolution",   offer_resolution_node)
    graph.add_node("confirm_resolution", confirm_resolution_node)
    graph.add_node("closing_negative",   closing_negative_node)
    graph.add_node("neutral_greeting",   neutral_greeting_node)
    graph.add_node("collect_feedback",   collect_feedback_node)
    graph.add_node("closing_neutral",    closing_neutral_node)
    graph.add_node("positive_greeting",  positive_greeting_node)
    graph.add_node("ask_review",         ask_review_node)
    graph.add_node("closing_positive",   closing_positive_node)
    graph.add_node("escalate",           escalate_node)

    # Entry point
    graph.set_entry_point("start")

    # Routing from start
    graph.add_conditional_edges("start", route_from_start, {
        "negative_greeting": "negative_greeting",
        "neutral_greeting":  "neutral_greeting",
        "positive_greeting": "positive_greeting",
    })

    # Negative flow
    graph.add_edge("negative_greeting",  "collect_order")
    graph.add_edge("collect_order",      "offer_resolution")
    graph.add_edge("offer_resolution",   "confirm_resolution")
    graph.add_edge("confirm_resolution", "closing_negative")
    graph.add_edge("closing_negative",   END)

    # Neutral flow
    graph.add_edge("neutral_greeting",  "collect_feedback")
    graph.add_edge("collect_feedback",  "closing_neutral")
    graph.add_edge("closing_neutral",   END)

    # Positive flow
    graph.add_edge("positive_greeting", "ask_review")
    graph.add_edge("ask_review",        "closing_positive")
    graph.add_edge("closing_positive",  END)

    # Escalation
    graph.add_edge("escalate", END)

    return graph.compile()

# ── Compiled graph instance ───────────────────────────────────
conversation_graph = build_graph()

# ── Public function called by api.py ─────────────────────────
def run_conversation(ticket_id: str, review_text: str,
                     sentiment: str, confidence: float) -> dict:
    """
    Run the full LangGraph conversation for a new review.
    Returns the initial bot messages to send to the customer.
    """
    initial_state: ConversationState = {
        "ticket_id":     ticket_id,
        "review_text":   review_text,
        "sentiment":     sentiment,
        "confidence":    confidence,
        "messages":      [],
        "step":          "start",
        "resolved":      False,
        "needs_agent":   False,
        "customer_name": None,
        "order_number":  None,
        "resolution":    None,
    }

    # Run only the first step (greeting) — rest happens turn by turn
    if sentiment == "Negative":
        state = negative_greeting_node(initial_state)
    elif sentiment == "Neutral":
        state = neutral_greeting_node(initial_state)
    else:
        state = positive_greeting_node(initial_state)

    return {
        "messages":     state["messages"],
        "step":         state["step"],
        "needs_agent":  state["needs_agent"],
        "resolved":     state["resolved"],
    }

def continue_conversation(ticket_id: str, review_text: str,
                          sentiment: str, current_step: str,
                          customer_message: str = "") -> dict:
    """
    Continue an existing conversation to the next step.
    Called when customer responds.
    """
    state: ConversationState = {
        "ticket_id":     ticket_id,
        "review_text":   review_text,
        "sentiment":     sentiment,
        "confidence":    0.9,
        "messages":      [],
        "step":          current_step,
        "resolved":      False,
        "needs_agent":   False,
        "customer_name": None,
        "order_number":  None,
        "resolution":    customer_message,
    }

    # Route to next step based on current step
    next_steps = {
        "collect_order":      collect_order_node,
        "offer_resolution":   offer_resolution_node,
        "confirm_resolution": confirm_resolution_node,
        "closing_negative":   closing_negative_node,
        "collect_feedback":   collect_feedback_node,
        "closing_neutral":    closing_neutral_node,
        "ask_review":         ask_review_node,
        "closing_positive":   closing_positive_node,
    }

    node_fn = next_steps.get(current_step)
    if node_fn:
        state = node_fn(state)
    else:
        state["resolved"] = True

    return {
        "messages":    state["messages"],
        "step":        state["step"],
        "needs_agent": state["needs_agent"],
        "resolved":    state["resolved"],
    }