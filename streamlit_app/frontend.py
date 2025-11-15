import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# --------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Arrhythmia Detector", layout="wide")
st.title("🫀 Intelligent Arrhythmia Classification Dashboard")
st.markdown("### Upload a 12-lead ECG signal to visualize and classify")

# --------------- ARRHYTHMIA LABELS ----------------
ARRHYTHMIA_CLASSES = {
    0: "Premature Ventricular Contraction",
    1: "Atrial Fibrillation",
    2: "Left Bundle Branch Block",
    3: "ST-Elavation",
    4: "Premature Atrio Ventricular Block",
    5: "Premature Atrial Contraction",
    6: "Normal Sinus Rhythm",
    7: "ST Depression",
    8: "Right Bundle Branch Block"
}

# --------------- FILE UPLOAD ----------------
uploaded = st.file_uploader("Upload ECG Signal (.csv or .npy)", type=["csv", "npy"])

if uploaded is not None:
    st.success(f"✅ File uploaded: {uploaded.name}")

    # ---- Parse ECG data ----
    if uploaded.name.endswith(".csv"):
        ecg_df = pd.read_csv(uploaded)
        ecg_data = ecg_df.to_numpy()
    else:
        ecg_data = np.load(uploaded)

    # Ensure correct shape
    if ecg_data.ndim == 1:
        ecg_data = np.expand_dims(ecg_data, axis=0)
    if ecg_data.shape[0] != 12:
        ecg_data = ecg_data.T if ecg_data.shape[1] == 12 else ecg_data

    # ---- ECG Plot ----
    st.markdown("### 📈 ECG Waveform Visualization")
    fig, axes = plt.subplots(6, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i in range(12):
        axes[i].plot(ecg_data[i][:1000])  # Plot first 1000 points for readability
        axes[i].set_title(f"Lead {i+1}")
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Amplitude")
    plt.tight_layout()
    st.pyplot(fig)

    # ---- Predict Button ----
    if st.button("🔍 Analyze ECG"):
        st.info("Sending data to model... please wait ⏳")

        # Convert data to bytes and send to backend
        with io.BytesIO() as buffer:
            np.save(buffer, ecg_data.astype(np.float32))
            buffer.seek(0)
            files = {"file": ("uploaded.npy", buffer.read())}
            response = requests.post("http://127.0.0.1:8000/predict", files=files)

        if response.status_code == 200:
            pred_idx = response.json()["prediction"]
            label = ARRHYTHMIA_CLASSES.get(pred_idx, f"Class {pred_idx}")

            st.success(f"🧠 Prediction: **{label}**")
            st.markdown("---")
            st.markdown("### ✅ Analysis Summary")
            st.info(
                f"""
                - **Predicted Class:** {label} 
                - **Lead Configuration:** 12 lead  
                """
            )
        else:
            st.error("❌ Error from backend: " + str(response.text))
