import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set experiment
mlflow.set_experiment("winequality-red-ci")

# End any active run sebelumnya
mlflow.end_run()

# Load dataset
df = pd.read_csv("winequality-red_preprocessing/train.csv")
df_test = pd.read_csv("winequality-red_preprocessing/test.csv")

X_train = df.drop(columns=["label"])
y_train = df["label"]

X_test = df_test.drop(columns=["label"])
y_test = df_test["label"]

# Train model
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)

    # Log metriks
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Simpan confusion matrix
    os.makedirs("artifacts", exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    cm_path = "artifacts/confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("Training selesai!")