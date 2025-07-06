import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import dagshub
dagshub.init(repo_owner='satyajeetrai007', repo_name='mlflow-dagshub', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/satyajeetrai007/mlflow-dagshub.mlflow/")

# Load data
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow experiment
mlflow.set_experiment('iris-rf-gridsearch')

# Define hyperparameter grid
param_grid = {
    'max_depth': [1, 2, 3, 4, 5],
    'n_estimators': [5, 10, 20, 50]
}

# Grid search
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
grid.fit(X_train, y_train)

# Best model
best_rf = grid.best_estimator_
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
best_params = grid.best_params_

with mlflow.start_run():


    for i in range(len(grid.cv_results_['params'])):
        with mlflow.start_run(nested = True, run_name = f"child_run_{i}"):
            
            mlflow.log_params(grid.cv_results_['params'][i])
            mlflow.log_metric('accuracy', grid.cv_results_['mean_test_score'][i])
        
    # Log best params & accuracy
    mlflow.log_params(best_params)
    mlflow.log_metric("best_accuracy", accuracy)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")

    mlflow.log_artifact("confusion_matrix.png")

    # Log model
    mlflow.sklearn.log_model(best_rf, "random_forest")

    # Log dataset as input
    train_data = pd.DataFrame(X_train, columns=iris.feature_names)
    train_data['target'] = y_train
    test_data = pd.DataFrame(X_test, columns=iris.feature_names)
    test_data['target'] = y_test
    mlflow.log_input(mlflow.data.from_pandas(train_data))
    mlflow.log_input(mlflow.data.from_pandas(test_data))

    # Set tags
    mlflow.set_tag("author", "satyajeet")
    mlflow.set_tag("model", "random_forest_gridsearch")

    print("Best params:", best_params)
    print("Best accuracy:", accuracy)
