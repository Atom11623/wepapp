from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import optuna

app = Flask(__name__)

# Function to run AutoML with hyperparameter tuning
def run_automl(data):
    # Split features and target
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define Optuna objective function
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 100, 500, step=50)
        max_depth = trial.suggest_int("max_depth", 3, 30, step=3)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10, step=2)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )

        model.fit(X_train, y_train)
        return model.score(X_test, y_test)

    # Run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    # Train best model
    best_params = study.best_params
    best_model = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        random_state=42
    )
    best_model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = best_model.score(X_test, y_test)

    return {
        "accuracy": accuracy,
        "best_params": best_params
    }

@app.route('/api/run-automl', methods=['POST'])
def run_pipeline():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    try:
        # Load dataset
        data = pd.read_csv(file)
        if 'Outcome' not in data.columns:
            return jsonify({"error": "Dataset must contain an 'Outcome' column."}), 400

        # Run AutoML pipeline
        results = run_automl(data)

        return jsonify({
            "message": "AutoML pipeline executed successfully.",
            "results": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
