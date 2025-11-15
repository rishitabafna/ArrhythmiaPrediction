step 1 :
    cd backend
    python -m uvicorn app:app --reload
    check at http://127.0.0.1:8000/
    http://127.0.0.1:8000/docs#/default/predict_predict_post

step 2 (in another terminal):
    cd streamlit_app
    streamlit run frontend.py
    check at http://localhost:8501


genarating a test case:
    import numpy as np
    x = np.random.rand(12, 5000).astype(np.float32)
    np.save("sample.npy", x)


if needed: 
    python -m venv venv
    venv\Scripts\activate 
    pip install -r requirements.txt


future work:
    Add probability bar chart for class confidence
    Deploy using Docker or Render
    Include Grad-CAM ECG interpretability
    Store predictions and patient history
    add as ai bot