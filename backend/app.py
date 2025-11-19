# backend/app.py

from fastapi import FastAPI, File, UploadFile
import uvicorn
import torch
import numpy as np
import io
from model_utils import load_model, predict_signal

app = FastAPI(title="Arrhythmia Classifier API")

DEVICE = "cpu"
model = load_model(DEVICE)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # -------------------- READ FILE --------------------
    if file.filename.endswith(".npy"):
        ecg = np.load(io.BytesIO(contents))
    else:
        import pandas as pd
        from io import StringIO
        df = pd.read_csv(StringIO(contents.decode()))
        ecg = df.to_numpy()

    # -------------------- FIX SHAPE --------------------
    # Expected: [1, 12, seq_len]
    if ecg.ndim == 2:
        ecg = np.expand_dims(ecg, axis=0)

    if ecg.shape[1] != 12 and ecg.shape[2] == 12:
        ecg = np.transpose(ecg, (0, 2, 1))

    ecg_tensor = torch.tensor(ecg, dtype=torch.float32)

    # -------------------- PREDICT --------------------
    pred_labels, probs = predict_signal(model, ecg_tensor, device=DEVICE)

    return {
        "predictions": pred_labels,
        "probabilities": probs
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)