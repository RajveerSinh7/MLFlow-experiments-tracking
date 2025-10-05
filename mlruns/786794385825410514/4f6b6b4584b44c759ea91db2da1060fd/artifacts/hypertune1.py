from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
import mlflow

#load the breast cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
Y = pd.Series(data.target, name='target')

#splitting into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

#Creating the randomforestclassifier model
rf = RandomForestClassifier(random_state=42)

#defining the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [10,50,100],
    'max_depth': [None,10,20,30]
}

#applying GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

#run without MLFlow from here
#grid_search.fit(X_train, Y_train)

#Displaying the best params and best score
#best_params = grid_search.best_params_
#best_score = grid_search.best_score_

#print(best_params)
#print(best_score)

mlflow.set_experiment('breast_cancer_rf_hp')

with mlflow.start_run() as parent:
    grid_search.fit(X_train,Y_train)

    #log all the child runs
    for i in range(len(grid_search.cv_results_['params'])):

        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(grid_search.cv_results_['params'][i])
            mlflow.log_metric("accuracy",grid_search.cv_results_["mean_test_score"][i])

    #displaying the best parameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    #log params
    mlflow.log_params(best_params)

    #log metrics
    mlflow.log_metric('accuracy',best_score)

    #log training data
    train_df = X_train.copy()
    train_df['target']=Y_train

    train_df = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df,"training")

    #log test data
    test_df = X_test.copy()
    test_df['target'] = Y_test

    test_df = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df,"testing")

    #log source code
    mlflow.log_artifact(__file__)

    #log the best model
    mlflow.sklearn.log_model(grid_search.best_estimator_,"random_forest")

    #set tags
    mlflow.set_tag("author","Rajveer Jadav")

    print(best_params)
    print(best_score)