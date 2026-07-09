# LensWord -- SQLite Database
# Stores all predictions, conversations, messages and tickets

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lensword.db')

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize all tables on startup"""
    conn = get_conn()
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP,
        ticket_id       TEXT UNIQUE,
        review_text     TEXT,
        sentiment       TEXT,
        confidence      REAL,
        priority        TEXT,
        action          TEXT,
        suggested_response TEXT,
        status          TEXT DEFAULT 'open',
        satisfaction    INTEGER DEFAULT NULL,
        session_id      TEXT
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS messages (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP,
        ticket_id   TEXT,
        sender      TEXT,
        message     TEXT,
        FOREIGN KEY (ticket_id) REFERENCES predictions(ticket_id)
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS alerts (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP,
        alert_type  TEXT,
        message     TEXT,
        resolved    INTEGER DEFAULT 0
    )''')

    conn.commit()
    conn.close()
    print("Database initialized!")

def generate_ticket_id():
    """Generate unique ticket ID like LW-2026-0001"""
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM predictions")
    count = c.fetchone()[0] + 1
    conn.close()
    return f"LW-{datetime.now().year}-{count:04d}"

def save_prediction(ticket_id, review_text, sentiment,
                    confidence, priority, action,
                    suggested_response, session_id=None):
    conn = get_conn()
    try:
        conn.execute('''INSERT INTO predictions
            (ticket_id, review_text, sentiment, confidence,
             priority, action, suggested_response, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (ticket_id, review_text, sentiment, confidence,
             priority, action, suggested_response, session_id))
        conn.commit()
    except Exception as e:
        print(f"DB save error: {e}")
    finally:
        conn.close()

def save_message(ticket_id, sender, message):
    conn = get_conn()
    try:
        conn.execute('''INSERT INTO messages
            (ticket_id, sender, message) VALUES (?, ?, ?)''',
            (ticket_id, sender, message))
        conn.commit()
    except Exception as e:
        print(f"DB message error: {e}")
    finally:
        conn.close()

def update_status(ticket_id, status):
    conn = get_conn()
    conn.execute("UPDATE predictions SET status=? WHERE ticket_id=?",
                 (status, ticket_id))
    conn.commit()
    conn.close()

def save_satisfaction(ticket_id, rating):
    conn = get_conn()
    conn.execute("UPDATE predictions SET satisfaction=? WHERE ticket_id=?",
                 (rating, ticket_id))
    conn.commit()
    conn.close()

def get_all_tickets(sentiment=None, priority=None, status=None, limit=50, offset=0):
    """Get tickets with optional filters and pagination"""
    conn = get_conn()

    # Count query for pagination
    count_query = "SELECT COUNT(*) FROM predictions WHERE 1=1"
    data_query  = "SELECT * FROM predictions WHERE 1=1"
    params = []

    if sentiment:
        count_query += " AND sentiment=?"
        data_query  += " AND sentiment=?"
        params.append(sentiment)
    if priority:
        count_query += " AND priority=?"
        data_query  += " AND priority=?"
        params.append(priority)
    if status:
        count_query += " AND status=?"
        data_query  += " AND status=?"
        params.append(status)

    total = conn.execute(count_query, params).fetchone()[0]

    data_query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    rows = conn.execute(data_query, params + [limit, offset]).fetchall()
    conn.close()

    return {
        "tickets": [dict(row) for row in rows],
        "total":   total,
        "limit":   limit,
        "offset":  offset,
        "pages":   (total + limit - 1) // limit
    }

def get_ticket(ticket_id):
    conn = get_conn()
    ticket = conn.execute(
        "SELECT * FROM predictions WHERE ticket_id=?",
        (ticket_id,)).fetchone()
    messages = conn.execute(
        "SELECT * FROM messages WHERE ticket_id=? ORDER BY timestamp ASC",
        (ticket_id,)).fetchall()
    conn.close()
    if not ticket:
        return None
    return {
        "ticket":   dict(ticket),
        "messages": [dict(m) for m in messages]
    }

def get_stats():
    conn = get_conn()

    counts = conn.execute('''
        SELECT sentiment, COUNT(*) as count
        FROM predictions GROUP BY sentiment
    ''').fetchall()

    today_counts = conn.execute('''
        SELECT sentiment, COUNT(*) as count
        FROM predictions
        WHERE DATE(timestamp) = DATE('now')
        GROUP BY sentiment
    ''').fetchall()

    trend = conn.execute('''
        SELECT DATE(timestamp) as date,
               sentiment, COUNT(*) as count
        FROM predictions
        WHERE timestamp >= DATE('now', '-7 days')
        GROUP BY DATE(timestamp), sentiment
        ORDER BY date ASC
    ''').fetchall()

    priority_counts = conn.execute('''
        SELECT priority, COUNT(*) as count
        FROM predictions GROUP BY priority
    ''').fetchall()

    open_count = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE status='open'"
    ).fetchone()[0]

    avg_satisfaction = conn.execute(
        "SELECT AVG(satisfaction) FROM predictions WHERE satisfaction IS NOT NULL"
    ).fetchone()[0]

    conn.close()

    return {
        "total_counts":    {row["sentiment"]: row["count"] for row in counts},
        "today_counts":    {row["sentiment"]: row["count"] for row in today_counts},
        "trend":           [dict(row) for row in trend],
        "priority_counts": {row["priority"]: row["count"] for row in priority_counts},
        "open_tickets":    open_count,
        "avg_satisfaction": round(avg_satisfaction, 1) if avg_satisfaction else None
    }

def check_drift():
    conn = get_conn()
    last_hour = conn.execute('''
        SELECT COUNT(*) FROM predictions
        WHERE timestamp >= DATETIME('now', '-1 hour')
        AND sentiment = 'Negative'
    ''').fetchone()[0]

    prev_hour = conn.execute('''
        SELECT COUNT(*) FROM predictions
        WHERE timestamp >= DATETIME('now', '-2 hours')
        AND timestamp < DATETIME('now', '-1 hour')
        AND sentiment = 'Negative'
    ''').fetchone()[0]
    conn.close()

    if prev_hour > 0:
        change = ((last_hour - prev_hour) / prev_hour) * 100
        if change >= 15:
            save_alert(
                "drift",
                f"Negative reviews up {change:.0f}% in the last hour ({last_hour} vs {prev_hour})"
            )

def save_alert(alert_type, message):
    conn = get_conn()
    conn.execute("INSERT INTO alerts (alert_type, message) VALUES (?, ?)",
                 (alert_type, message))
    conn.commit()
    conn.close()

def get_alerts():
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM alerts WHERE resolved=0 ORDER BY timestamp DESC LIMIT 10"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]