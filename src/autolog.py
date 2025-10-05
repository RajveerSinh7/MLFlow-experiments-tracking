import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
wine = load_wine()
X = wine.data
Y = wine.target

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)

# Hyperparameters
max_depth = 11
n_estimators = 10

# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("exp1")

# Enable autologging for sklearn
mlflow.sklearn.autolog()

# Start MLflow run
with mlflow.start_run(run_name=f"run_depth_{max_depth}"):
    # Train model
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, Y_train)

    # Predictions
    y_pred = rf.predict(X_test)

    # Create and save confusion matrix plot
    cm = confusion_matrix(Y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("Confusionmatrix.png")
    plt.close()

    # Log the confusion matrix as artifact
    mlflow.log_artifact("Confusionmatrix.png")