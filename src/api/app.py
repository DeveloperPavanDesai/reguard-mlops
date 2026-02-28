import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from src.model.model import ANN

app = FastAPI(title="Reguard ML Inference API")

# Load model
model = ANN(dropout_rate=0.0)
model.load_state_dict(torch.load("models/mnist_model.pth", map_location=torch.device("cpu")))
model.eval()


class InputData(BaseModel):
    image: list 


@app.get("/")
def home():
    return {"message": "Reguard ML API is running"}


@app.post("/predict")
def predict(data: InputData):

    image = np.array(data.image).reshape(1, 1, 28, 28)
    tensor = torch.tensor(image, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()

    return {
        "predicted_digit": prediction,
        "confidence": float(torch.max(probabilities))
    }