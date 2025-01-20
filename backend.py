import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from fastapi import FastAPI, File, UploadFile
import uvicorn
import shutil
from pycaret.classification import setup, compare_models, pull

app = FastAPI()

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Save the uploaded file
    with open(file.filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load the dataset
    data = pd.read_csv(file.filename)

    # Split the dataset
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Hyperparameter Tuning for Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    grid_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=3, scoring='roc_auc')
    grid_rf.fit(X_train, y_train)
    optimized_rf_model = grid_rf.best_estimator_

    # Hyperparameter Tuning for XGBoost
    xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
    param_grid_xgb = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    grid_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, cv=3, scoring='roc_auc')
    grid_xgb.fit(X_train, y_train)
    optimized_xgb_model = grid_xgb.best_estimator_

    # Model Evaluation
    rf_metrics = {
        'Accuracy': accuracy_score(y_test, optimized_rf_model.predict(X_test)),
        'Precision': precision_score(y_test, optimized_rf_model.predict(X_test)),
        'Recall': recall_score(y_test, optimized_rf_model.predict(X_test)),
        'F1 Score': f1_score(y_test, optimized_rf_model.predict(X_test)),
        'ROC-AUC': roc_auc_score(y_test, optimized_rf_model.predict_proba(X_test)[:, 1])
    }

    xgb_metrics = {
        'Accuracy': accuracy_score(y_test, optimized_xgb_model.predict(X_test)),
        'Precision': precision_score(y_test, optimized_xgb_model.predict(X_test)),
        'Recall': recall_score(y_test, optimized_xgb_model.predict(X_test)),
        'F1 Score': f1_score(y_test, optimized_xgb_model.predict(X_test)),
        'ROC-AUC': roc_auc_score(y_test, optimized_xgb_model.predict_proba(X_test)[:, 1])
    }

    # Leaderboard using PyCaret
    s = setup(data, target="Outcome", silent=True, verbose=False)
    best_model = compare_models()
    leaderboard = pull()

    return {
        "best_rf_params": grid_rf.best_params_,
        "best_xgb_params": grid_xgb.best_params_,
        "rf_metrics": rf_metrics,
        "xgb_metrics": xgb_metrics,
        "pycaret_leaderboard": leaderboard.to_dict(),
        "pycaret_best_model": str(best_model)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
