import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

# models/fcnn/train.py
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from preprocessing import load_dataset, split_train_test, prepare_features
from models.fcnn.model import FCNN
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

MODEL_PATH = "models/fcnn/fcnn_model.pt"
SCALER_PATH = "models/fcnn/scaler.pkl"

def train_fcnn(epochs=100, patience=10):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    df = load_dataset("data/features/dataset.csv")
    train_df, test_df = split_train_test(df)

    X_train, y_train, scaler = prepare_features(train_df, fit=True)
    X_test, y_test, _ = prepare_features(test_df, scaler=scaler, fit=False)

    # Save scaler
    joblib.dump(scaler, SCALER_PATH)

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    # Class weights (IMPORTANT)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_df["anomaly"].values
    )
    pos_weight = torch.tensor(class_weights[1] / class_weights[0]).to(device)

    # Model
    model = FCNN(input_dim=18).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test)

        scheduler.step(val_loss)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {loss.item():.4f} "
            f"Val Loss: {val_loss.item():.4f}"
        )

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("[!] Early stopping triggered.")
                break

    print("[✓] FCNN training completed. Best model saved.")

if __name__ == "__main__":
    train_fcnn()