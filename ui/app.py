# ui/app.py

import streamlit as st
import numpy as np
import os
import sys
from tensorflow.keras.models import load_model

# 📍 Add parent folder to path so we can import from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.extract_features import extract_features

# ✅ Load the trained CNN model
MODEL_PATH = r"C:\Users\VICTUS\OneDrive\Desktop\respiratory_disease_classification\notebooks\best_cnn_model.h5"
model = load_model(MODEL_PATH)

# ✅ Class labels from your dataset (in training order)
CLASS_LABELS = ['COPD', 'Healthy', 'Pneumonia', 'URTI']

# 🎨 Streamlit UI setup
st.set_page_config(page_title="🩺 Respiratory Disease Classifier", layout="centered")
st.title("🫁 Respiratory Disease Detection from Cough Sound")
st.markdown("Upload a `.wav` lung sound file to detect the disease type using a CNN model.")

# 📤 Upload section
uploaded_file = st.file_uploader("📤 Upload your lung audio (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    with st.spinner("🔬 Extracting features..."):
        features = extract_features(uploaded_file)

    if features is not None:
        features = np.expand_dims(features, axis=0)  # (1, 216, 40, 1)

        with st.spinner("🧠 Making prediction..."):
            prediction = model.predict(features)[0]
            predicted_class = CLASS_LABELS[np.argmax(prediction)]
            confidence = np.max(prediction)

        # 🎯 Styled Output
        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")
        st.markdown(
            f"""
            <div style='text-align: center; font-size: 28px; font-weight: bold;'>
                🩻 <span style='color: #2E86AB;'>{predicted_class}</span><br>
                🔬 Confidence: <span style='color: #28B463;'>{confidence*100:.2f}%</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("---")

    else:
        st.error("❌ Feature extraction failed. Please upload a different audio file.")
