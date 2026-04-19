import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
data = pd.read_csv("creditcard.csv")

# Take only few features (simplify for now)
X = data[['Amount', 'Time']]
y = data['Class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

# Train model
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, "fraud_model.pkl")

print("Model trained and saved!")

print("Model trained and saved!")