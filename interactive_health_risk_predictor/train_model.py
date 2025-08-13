import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# --------------------------------------------------
# Locate base directory of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
DATA_PATH = os.path.join(BASE_DIR, "data", "diabetes.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# --------------------------------------------------
# Load dataset
df = pd.read_csv(DATA_PATH)

# Features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save scaler
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

# Save model with extra info
model_data = {
    "pipeline": model,
    "feature_names": list(X.columns)
}
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model_data, f)

print("✅ Model and scaler saved successfully!")
