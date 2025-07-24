import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# === Settings ===
st.set_page_config(page_title="🧠 Brain Tumor Detector", layout="centered")

# === Title ===
st.markdown("<h1 style='text-align: center; color: teal;'>🧠 Brain Tumor MRI Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an MRI scan and see predictions from multiple models</p>", unsafe_allow_html=True)

# === Class Names (based on your dataset folders) ===
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# === Load Models Dynamically ===
@st.cache_resource
def load_all_models():
    models = {}
    model_dir = "models"
    for file in os.listdir(model_dir):
        if file.endswith(".h5"):
            model_name = os.path.splitext(file)[0].replace("_", " ").title()
            models[model_name] = load_model(os.path.join(model_dir, file))
    return models

# === Preprocess Image ===
def preprocess(img):
    img = img.resize((224, 224)).convert('RGB')
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# === Plot Confidence Bar Chart ===
def plot_confidences(confidences):
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(class_names, confidences, color='skyblue')
    ax.set_ylim(0, 1)
    ax.set_ylabel("Confidence")
    ax.set_title("Model Confidence per Class")
    for bar, val in zip(bars, confidences):
        ax.text(bar.get_x() + bar.get_width()/2.0, val + 0.01, f"{val*100:.1f}%", ha='center')
    st.pyplot(fig)

# === Upload UI ===
uploaded_file = st.file_uploader("📤 Upload an MRI image (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)

    st.markdown("---")
    st.image(img, caption="🖼️ Uploaded MRI Image", use_column_width=True)
    st.markdown("🔄 **Running predictions with all models...**")

    # Load and preprocess
    input_data = preprocess(img)
    models = load_all_models()

    # Predictions
    st.markdown("## 🔍 Predictions")
    for model_name, model in models.items():
        with st.spinner(f"Predicting with {model_name}..."):
            pred = model.predict(input_data)[0]
            pred_class = class_names[np.argmax(pred)]
            confidence = np.max(pred)

            st.markdown(f"### ✅ `{model_name}`")
            st.success(f"🧠 Predicted: **{pred_class}** with **{confidence*100:.2f}%** confidence")
            plot_confidences(pred)
            st.markdown("---")
