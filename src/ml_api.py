from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import asyncio
import os
from pathlib import Path
import json

app = FastAPI()

# Tunnel (optional)
TUNNEL_URL = os.environ.get("PUBLISHED_URL") or os.environ.get("TUNNEL_URL", "")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ FIXED DB PATH (IMPORTANT)
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "tube_detections.db"

print("📦 Reading DB at:", DB_PATH)

# ✅ CREATE TABLE IF NOT EXISTS
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tube_detections (
            tube_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            brand_name TEXT,
            confidence REAL,
            track_id INTEGER
        )
    """)
    conn.commit()
    conn.close()

init_db()

def get_stats():
    total_tubs = 0
    company_counts = {}
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # ✅ CORRECT TABLE
        cursor.execute("SELECT COUNT(*) FROM tube_detections")
        total_tubs = cursor.fetchone()[0]
        
        cursor.execute("SELECT brand_name, COUNT(*) FROM tube_detections GROUP BY brand_name")
        rows = cursor.fetchall()
        
        for brand, count in rows:
            company_counts[brand] = count
            
        conn.close()
    except Exception as e:
        print("DB read error:", e)
            
    return {"total_tubs": total_tubs, "company_counts": company_counts}


@app.get("/api/info")
def api_info():
    return {
        "local_url": "http://localhost:8001",
        "tunnel_url": TUNNEL_URL or None,
        "docs": "http://localhost:8001/docs",
        "stats": "/api/stats",
        "ws_stats": "ws://localhost:8001/ws/stats",
    }


@app.get("/api/stats")
def read_stats():
    stats = get_stats()
    print("📊 Stats:", stats)
    return stats


@app.websocket("/ws/stats")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    last_count = -1
    try:
        while True:
            stats = get_stats()
            if stats["total_tubs"] != last_count:
                await websocket.send_text(json.dumps(stats))
                last_count = stats["total_tubs"]
            await asyncio.sleep(1)
    except:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)