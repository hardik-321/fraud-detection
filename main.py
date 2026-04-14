from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import pandas as pd
import numpy as np
import sqlite3

conn = sqlite3.connect("transactions.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS transactions (
    amount REAL,
    time REAL,
    type TEXT,
    fraud INTEGER,
    confidence REAL
)
""")

conn.commit()

from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# CREATE SIMPLE DATASET
# -------------------------
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
def predict(data: Transaction):
    amount = data.amount
    time = data.time
    type = data.type

    # Simple logic (you can keep your AI logic here)
    if amount > 1000000:
        return {
            "fraud": True,
            "confidence": 90,
            "risk": "High Risk"
        }

    elif amount > 100000:
        return {
            "fraud": True,
            "confidence": 70,
            "risk": "Medium Risk"
        }

    elif time < 5 and amount > 50000:
        return {
            "fraud": True,
            "confidence": 65,
            "risk": "Medium Risk"
        }

    else:
        return {
            "fraud": False,
            "confidence": 30,
            "risk": "Low Risk"
        }

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