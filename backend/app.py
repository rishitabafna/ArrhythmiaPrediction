# backend/app.py
from fastapi import FastAPI, File, UploadFile
import uvicorn
import torch
import numpy as np
import io
from model_utils import load_model, predict_signal

app = FastAPI(title="Arrhythmia Classifier")

device = "cpu"
model = load_model(device)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded ECG data (expects .npy or .csv file)
    contents = await file.read()
    
    if file.filename.endswith(".npy"):
        ecg = np.load(io.BytesIO(contents))
    else:
        import pandas as pd
        from io import StringIO
        df = pd.read_csv(StringIO(contents.decode()))
        ecg = df.to_numpy()

    # Make sure it’s in [batch, channels, sequence_length]
    if ecg.ndim == 2:
        ecg = np.expand_dims(ecg, axis=0)
    ecg_tensor = torch.tensor(ecg, dtype=torch.float32)

    pred = predict_signal(model, ecg_tensor, device=device)
    return {"prediction": int(pred)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
