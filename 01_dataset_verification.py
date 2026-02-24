from preprocessing.feature_loader import load_dataset, split_train_test
from preprocessing.feature_loader import prepare_features
from sklearn.preprocessing import StandardScaler

df = load_dataset("data/features/dataset.csv")
train_df, test_df = split_train_test(df)

print("Total records:", len(df))
print("Train size:", len(train_df))
print("Test size:", len(test_df))

print("\nLabel distribution (Train):")
print(train_df["anomaly"].value_counts())

print("\nLabel distribution (Test):")
print(test_df["anomaly"].value_counts())

scaler = StandardScaler()

X_train, y_train, scaler = prepare_features(train_df, scaler, fit=True)
X_test, y_test, _ = prepare_features(test_df, scaler, fit=False)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)