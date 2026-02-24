import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import joblib
from sklearn.metrics import (
    classification_report,
    matthews_corrcoef,
    average_precision_score,
    roc_auc_score,
    confusion_matrix
)
from preprocessing import load_dataset, split_train_test, prepare_features

MODEL_PATH = "models/adaboost/adaboost_model.pkl"
SCALER_PATH = "models/adaboost/scaler.pkl"


def evaluate_adaboost():
    # Load model and scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Load dataset
    df = load_dataset("data/features/dataset.csv")
    _, test_df = split_train_test(df)

    # Prepare test features
    X_test, y_test, _ = prepare_features(test_df, scaler=scaler, fit=False)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    print("\n=== AdaBoost Evaluation Results ===\n")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nMCC:", matthews_corrcoef(y_test, y_pred))
    print("AUCPR:", average_precision_score(y_test, y_prob))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))


if __name__ == "__main__":
    evaluate_adaboost()