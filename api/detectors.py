import torch
import joblib
import numpy as np
from models.fcnn.model import FCNN

MODEL_PATH = "models/fcnn/fcnn_model.pt"
SCALER_PATH = "models/fcnn/scaler.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model once
model = FCNN(input_dim=18).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

scaler = joblib.load(SCALER_PATH)


def detect_anomaly(features):
    X = np.array(features).reshape(1, -1)
    X = scaler.transform(X)
    X = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        logit = model(X)
        prob = torch.sigmoid(logit).item()

    return prob, prob >= 0.5