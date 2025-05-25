from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal, Dict
import joblib
import numpy as np
import uvicorn
import traceback

# Load all models
models = {
    ("classification", "goals"): joblib.load("goals_classification_model_XGB.pkl"),
    ("regression", "goals"): joblib.load("goals_regression_model_XGB.pkl"),
    ("classification", "assists"): joblib.load("random_forest_assists_clf.pkl"),
    ("regression", "assists"): joblib.load("random_forest_assists_reg.pkl")
}

# Updated list of 34 numeric features used by the models (excluding targets)
FEATURE_NAMES = [
    'ID', 'position', 'team', 'saves_per_90', 'penalties_saved', 'own_goals', 'clean_sheets',
    'saves', 'yellow_cards', 'red_cards', 'Wins', 'Losses', 'Goals Conceded', 'Tackles',
    'Tackle success %', 'Interceptions', 'Recoveries', 'Duels won', 'Duels lost',
    'Aerial battles won', 'Aerial battles lost', 'Passes', 'Big Chances Created',
    'Cross accuracy %', 'Accurate long balls', 'Fouls', 'Headed goals', 'Shots',
    'Shots on target', 'Shooting accuracy %', 'Big chances missed', 'Saves',
    'Goals_Assists', 'Total_Contributions', 'Minutes_per_Goal', 'Minutes_per_Assist'
]


class PredictionRequest(BaseModel):
    model_type: Literal["classification", "regression"]
    target: Literal["goals", "assists"]
    features: Dict[str, float]

app = FastAPI()

@app.post("/predict")
def predict(request: PredictionRequest):
    key = (request.model_type, request.target)
    if key not in models:
        raise HTTPException(status_code=400, detail="Invalid model type or target")

    model = models[key]

    print("\n--- DEBUG ---")
    print("Features received:", list(request.features.keys()))
    print("Expected features:", FEATURE_NAMES)

    missing = [f for f in FEATURE_NAMES if f not in request.features]
    if missing:
        print(f"Missing features filled with 0.0: {missing}")
    input_data = [request.features.get(feat, 0.0) for feat in FEATURE_NAMES]
    input_array = np.array(input_data).reshape(1, -1)

    try:
        print("Input shape:", input_array.shape)
        prediction = model.predict(input_array)
        print("Prediction raw output:", prediction)
        return {"prediction": prediction[0].item() if hasattr(prediction[0], 'item') else float(prediction[0])}
    except Exception as e:
        print("Error during prediction:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Uncomment to run locally
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
