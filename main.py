from fastapi import FastAPI
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import pandas as pd
import numpy as np
import sqlite3

conn = sqlite3.connect("transactions.db", check_same_thread=False)
cursor = conn.cursor()


cursor.execute("""
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    amount REAL,
    time REAL,
    type TEXT,
    fraud BOOLEAN,
    confidence REAL,
    risk TEXT
)
""")

conn.commit()

from sklearn.ensemble import RandomForestClassifier

app = FastAPI()
model = joblib.load("fraud_model.pkl")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data = []

for _ in range(5000):
    amount = np.random.randint(100, 1000000)
    time = np.random.randint(0, 24)
    type_val = np.random.choice(["UPI", "Card", "International"])

    fraud = 0
    if amount > 20000 and time > 20:
        fraud = 1
    if type_val == "International" and amount > 10000:
        fraud = 1

    data.append([amount, time, type_val, fraud])

# Add extreme fraud cases
for _ in range(500):
    amount = np.random.randint(1000000, 10000000)
    time = np.random.randint(0, 24)
    type_val = "International"

    data.append([amount, time, type_val, 1])

df = pd.DataFrame(data, columns=["amount", "time", "type", "fraud"])

df["type"] = df["type"].map({"UPI": 0, "Card": 1, "International": 2})

X = df[["amount", "time", "type"]]
y = df["fraud"]
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier()
model.fit(X_scaled, y)

from pydantic import BaseModel

class Transaction(BaseModel):
    amount: float
    time: float
    type: str

@app.post("/predict")
def predict(data: dict):
    try:
        amount = float(data.get("amount"))
        time = float(data.get("time"))
        type_ = data.get("type")

      # Realistic fraud scoring system

        score = 0

        # Amount scoring
        if amount > 100000:
            score += 40
        elif amount > 50000:
            score += 25
        elif amount > 20000:
            score += 10

        # Time / frequency scoring
        if time > 200:
            score += 40
        elif time > 100:
            score += 25
        elif time > 50:
            score += 10

        # Combined behavior
        if amount < 1000 and time > 100:
            score += 30

        if amount > 50000 and time < 10:
            score += 20

        # Final decision (ONLY fraud boolean for your DB)
        if score >= 50:
            fraud = True
        else:
            fraud = False

        # Save to DB
        cursor.execute(
            "INSERT INTO transactions (amount, time, type, fraud) VALUES (?, ?, ?, ?)",
            (amount, time, type_, fraud)
        )
        
        conn.commit()

        return {
            "fraud": fraud
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/history")
def get_history():
    cursor.execute("SELECT * FROM transactions")
    rows = cursor.fetchall()

    data = []
    for row in rows:
        data.append({
            "amount": row[0],
            "time": row[1],
            "type": row[2],
            "fraud": bool(row[3]),
            "confidence": row[4]
        })

    return data

@app.get("/")
def home():
    return FileResponse(os.path.join(os.getcwd(), "index.html"))

@app.get("/history")
def get_history():
    conn = sqlite3.connect("transactions.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM transactions ORDER BY id DESC")
    rows = cursor.fetchall()

    data = []
    for row in rows:
        data.append({
            "amount": row[1],
            "time": row[2],
            "type": row[3],
            "fraud": row[4],
            "confidence": row[5],
            "risk": row[6]
        })

    conn.close()
    return data