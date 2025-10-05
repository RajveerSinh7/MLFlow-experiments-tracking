import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#load wine dataset
wine = load_wine()
X = wine.data
Y = wine.target

#Train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)

#Define the params for RF model
max_depth =11
n_estimators = 10

# Set MLflow to use local folder for tracking
mlflow.set_tracking_uri("file:///Users/rajveersinhjadav/Desktop/MLflow/mlruns")
#mlflow.set_experiment("Wine_RF_Experiment")
mlflow.set_experiment("exp1")

#ML flow (everything inside this block is logged as 1 experiment run)
with mlflow.start_run(run_name=f"run_depth_{max_depth}", nested=False):
    #train the model
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train,Y_train)

    #make predictions and evaluate
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(Y_test,y_pred)

    #log results to mlflow
    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max_depth',max_depth)
    mlflow.log_param('n_estimators',n_estimators)

    #creating a confusion matrix plot
    cm = confusion_matrix(Y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d',cmap='Blues', xticklabels=wine.target_names,yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    #save plot
    plt.savefig("Confusionmatrix.png")

    #log the artifacts using mlflow
    mlflow.log_artifact("Confusionmatrix.png")
    mlflow.log_artifact(__file__)

    print(accuracy)