import sqlite3
import os
from datetime import datetime

DB_PATH = "safevision.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            event_type TEXT,
            message TEXT,
            file_path TEXT,
            model_accuracy REAL
        )
    ''')
    conn.commit()
    conn.close()

def log_event(event_type, message, file_path="", model_accuracy=0.0):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO logs (timestamp, event_type, message, file_path, model_accuracy)
        VALUES (?, ?, ?, ?, ?)
    ''', (timestamp, event_type, message, file_path, model_accuracy))
    conn.commit()
    conn.close()

def get_all_logs():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT timestamp, event_type, message, file_path, model_accuracy FROM logs ORDER BY id DESC')
    rows = cursor.fetchall()
    conn.close()
    return rows

init_db()
