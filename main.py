import warnings

warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import math
from typing import List
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
import os

load_dotenv()

# PostgreSQL Connection URL
SCALER = int(os.getenv("MINMAX_SCALER", 229426))
WINDOWSIZE = int(os.getenv("WINDOWSIZE", 10))
PODS_MIN = int(os.getenv("PODS_MIN", 10))
RRS = float(os.getenv("RRS", 0.6))
WORKLOAD_POD = int(os.getenv("WORKLOAD_POD", 300))

model = load_model("BiLSTM_autoscaling_ep20.keras")

app = FastAPI()


# Define the request schema
class PredictionRequest(BaseModel):
    previous_workload: List[float]
    current_pods: int


@app.post("/api/predict")
def predict(request: PredictionRequest):
    # Ensure the input is valid
    if len(request.previous_workload) < WINDOWSIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Input array must be at least {WINDOWSIZE}",
        )

    # Call the make_prediction function
    predicted_workload, predicted_pods = make_prediction(
        model=model,
        previous_workload=np.array(request.previous_workload),
        current_pods=request.current_pods,
    )
    return JSONResponse(
        status_code=200,
        content={
            "predicted_workload": float(predicted_workload),
            "predicted_pods": predicted_pods,
        },
    )


# The make_prediction function from your provided code
def make_prediction(model, previous_workload, current_pods):
    """
    Make a prediction from a given input array using a trained model.

    Parameters:
    - model: The trained LSTM model.
    - input_array: A 1D numpy array of new requests to make a prediction on.
    - sequence_length: The length of the input sequence for the model.

    Returns:
    - prediction: The predicted value after reverse scaling.
    """
    # Normalize the input data
    previous_workload_scaled = previous_workload / SCALER

    # Create a sequence for prediction (sequence_length-sized window)
    input_sequence = []
    input_sequence.append(previous_workload_scaled[-WINDOWSIZE:])

    input_sequence = np.array(input_sequence)

    # Reshape for LSTM (samples, timesteps, features)
    input_sequence = input_sequence.reshape(
        (input_sequence.shape[0], input_sequence.shape[1], 1)
    )

    # Make predictions
    prediction_scaled = model.predict(input_sequence)

    # Reverse the scaling (denormalize the prediction)
    predicted_workload = prediction_scaled * SCALER

    predicted_pods = pods_adaption(predicted_workload, current_pods)

    return predicted_workload, predicted_pods


def pods_adaption(predicted_workload, current_pods):

    pods_t1 = predicted_workload / WORKLOAD_POD

    if pods_t1 > current_pods:
        return math.ceil(pods_t1)

    elif pods_t1 < current_pods:
        pods_t1 = max(pods_t1, PODS_MIN)
        pods_surplus = (current_pods - pods_t1) * RRS
        return math.ceil(max((current_pods - pods_surplus), PODS_MIN))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8081)
