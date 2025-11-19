# frontend.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Arrhythmia Detector", layout="wide")
st.title("🫀 Intelligent Arrhythmia Classification Dashboard")
st.markdown("### Upload a 12-lead ECG signal to visualize and classify")

ARRHYTHMIA_CLASSES = [
    "Premature Ventricular Contraction",
    "Atrial Fibrillation",
    "Left Bundle Branch Block",
    "ST-Elevation",
    "I-Degree AtrioVentricular Block",
    "Premature Atrial Contraction",
    "Normal Sinus Rhythm",
    "ST Depression",
    "Right Bundle Branch Block"
]

uploaded = st.file_uploader("Upload ECG Signal (.csv or .npy)", type=["csv", "npy"])

if uploaded:
    st.success(f"File uploaded: {uploaded.name}")

    # -------------------- LOAD ECG --------------------
    if uploaded.name.endswith(".csv"):
        ecg_df = pd.read_csv(uploaded)
        ecg_data = ecg_df.to_numpy()
    else:
        ecg_data = np.load(uploaded)

    if ecg_data.ndim == 1:
        ecg_data = ecg_data.reshape(12, -1)

    if ecg_data.shape[0] != 12 and ecg_data.shape[1] == 12:
        ecg_data = ecg_data.T

    # -------------------- ECG PLOTS --------------------
    st.markdown("### 📈 ECG Waveform Visualization")

    fig, axes = plt.subplots(6, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i in range(12):
        axes[i].plot(ecg_data[i][:1000])
        axes[i].set_title(f"Lead {i + 1}")
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Amplitude")

    st.pyplot(fig)

    # -------------------- PREDICT --------------------
    if st.button("🔍 Analyze ECG"):
        st.info("Running model... please wait ⏳")

        with io.BytesIO() as buffer:
            np.save(buffer, ecg_data.astype(np.float32))
            buffer.seek(0)
            files = {"file": ("ecg.npy", buffer.read())}
            response = requests.post("http://127.0.0.1:8000/predict", files=files)

        if response.status_code != 200:
            st.error("❌ Backend error: " + response.text)
        else:
            result = response.json()
            preds = result["predictions"]   # ignore probabilities

            # -------------------- DISPLAY ONLY LABELS --------------------
            st.success("🧠 Prediction Complete!")

            st.markdown("## ✅ Detected Arrhythmias")

            detected = [ARRHYTHMIA_CLASSES[i] for i, p in enumerate(preds) if p == 1]

            if len(detected) == 0:
                st.info("No arrhythmias detected.")
            else:
                for label in detected:
                    st.markdown(f"### 🔹 {label}")