import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

# models/fcnn/evaluate.py
import torch
import joblib
import numpy as np
from sklearn.metrics import (
    classification_report,
    matthews_corrcoef,
    average_precision_score,
    roc_auc_score,
    confusion_matrix
)
from preprocessing import load_dataset, split_train_test, prepare_features
from models.fcnn.model import FCNN

MODEL_PATH = "models/fcnn/fcnn_model.pt"
SCALER_PATH = "models/fcnn/scaler.pkl"

def evaluate_fcnn():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = FCNN(input_dim=18).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    scaler = joblib.load(SCALER_PATH)

    # Load data
    df = load_dataset("data/features/dataset.csv")
    _, test_df = split_train_test(df)

    X_test, y_test, _ = prepare_features(test_df, scaler=scaler, fit=False)

    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_test)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()

    y_pred = (probs >= 0.5).astype(int)

    print("\n=== FCNN Evaluation Results ===\n")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("MCC:", matthews_corrcoef(y_test, y_pred))
    print("AUCPR:", average_precision_score(y_test, probs))
    print("ROC-AUC:", roc_auc_score(y_test, probs))


if __name__ == "__main__":
    evaluate_fcnn()