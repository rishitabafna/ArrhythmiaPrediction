import numpy as np
import torch
from model_utils import load_model, predict_signal

# Load model
device = torch.device("cpu")
model = load_model(device)

# Load ECG file (csv or npy)
file_path = "single_signal.npy"   # change to your file
if file_path.endswith(".npy"):
    ecg = np.load(file_path)
else:
    import pandas as pd
    ecg = pd.read_csv(file_path).to_numpy()

# Prepare input
if ecg.ndim == 1:
    ecg = np.expand_dims(ecg, axis=0)
if ecg.shape[0] != 12:
    ecg = ecg.T

ecg_tensor = torch.tensor(ecg, dtype=torch.float32).unsqueeze(0)

# Predict
pred = predict_signal(model, ecg_tensor, device=device)
print("Prediction:", pred)
