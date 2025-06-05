import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import seaborn as sns
import pandas as pd

# Streamlit UI setup
st.set_page_config(
    page_title="Rice Leaf Disease Classification", layout="centered")

st.title("🌾 Rice Leaf Disease Classification")
st.write("Upload a rice leaf image, and it will be classified using four different models for comparison.")

# Model paths
MODEL_PATHS = {
    "Xception": "../results/xception_model.h5",
    "VGG16": "../results/vgg16_model.h5",
    "MobileNet": "../results/mobilenet_model.h5",
    "EfficientNetB0": "../results/efficientnet_model.h5"
}

# Load models
@st.cache(allow_output_mutation=True)
def load_models():
    models = {}
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            models[name] = load_model(path)
    return models

models = load_models()

# Upload image
uploaded_file = st.file_uploader("Upload a rice leaf image:", type=["jpg", "png", "jpeg"])

# Classification
if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    st.image(img, caption="Uploaded Image", use_column_width=True)

    st.subheader("Model Predictions:")
    class_labels = ["Bacterialblight", "Blast", "Brownspot", "Tungro"]

    col1, col2 = st.columns(2)
    with col1:
        for name in list(models.keys())[:2]:
            pred = models[name].predict(img_array)
            predicted_class = class_labels[np.argmax(pred)]
            st.success(f"{name}: {predicted_class}")
    with col2:
        for name in list(models.keys())[2:]:
            pred = models[name].predict(img_array)
            predicted_class = class_labels[np.argmax(pred)]
            st.success(f"{name}: {predicted_class}")

# Show comparison graphs
if st.checkbox("Show model comparison graphs"):
    scores_path = "../results/metrics.json"
    if os.path.exists(scores_path):
        with open(scores_path, "r") as file:
            scores = json.load(file)

        labels = list(scores.keys())
        acc = [scores[m]["accuracy"] for m in labels]
        f1 = [scores[m]["f1_score"] for m in labels]
        recall = [scores[m]["recall"] for m in labels]
        precision = [scores[m]["precision"] for m in labels]

        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(labels))
        ax.bar(x - 0.3, acc, width=0.15, label='Accuracy')
        ax.bar(x - 0.1, precision, width=0.15, label='Precision')
        ax.bar(x + 0.1, recall, width=0.15, label='Recall')
        ax.bar(x + 0.3, f1, width=0.15, label='F1-Score')

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison")
        ax.legend()
        st.pyplot(fig)

        # Table of metrics
        st.subheader("Evaluation Table")
        table_data = {
            "Model": labels,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        }
        df = pd.DataFrame(table_data)
        st.dataframe(df)

        # Download as CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Metrics as CSV", csv, "model_metrics.csv", "text/csv")
    else:
        st.warning("model_scores.json not found.")

# Show confusion matrices
if st.checkbox("Show confusion matrices"):
    cm_path = "../results/confusion_matrices.json"
    if os.path.exists(cm_path):
        with open(cm_path, "r") as file:
            cm_data = json.load(file)

        class_labels = ["Bacterialblight", "Blast", "Brownspot", "Tungro"]
        for model_name, matrix in cm_data.items():
            st.subheader(f"Confusion Matrix - {model_name}")
            df_cm = pd.DataFrame(matrix, index=class_labels, columns=class_labels)
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
            st.pyplot(fig_cm)
    else:
        st.warning("confusion_matrices.json not found.")
