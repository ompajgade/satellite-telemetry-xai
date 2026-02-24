import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import joblib
from sklearn.ensemble import AdaBoostClassifier
from preprocessing import load_dataset, split_train_test, prepare_features

MODEL_PATH = "models/adaboost/adaboost_model.pkl"
SCALER_PATH = "models/adaboost/scaler.pkl"


def train_adaboost():
    # Load dataset
    df = load_dataset("data/features/dataset.csv")
    train_df, test_df = split_train_test(df)

    # Prepare features
    X_train, y_train, scaler = prepare_features(train_df, fit=True)

    # AdaBoost baseline model
    model = AdaBoostClassifier(
        n_estimators=200,
        learning_rate=0.5,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Save model and scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("[✓] AdaBoost model trained and saved.")
    return model, scaler


if __name__ == "__main__":
    train_adaboost()