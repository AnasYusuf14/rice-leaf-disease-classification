import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# UI setup
st.set_page_config(
    page_title="Rice Leaf Disease Classification", layout="centered")

st.title("🌾 Rice Leaf Disease Classification")
st.write("Upload a rice leaf image and classify it using the selected trained model.")

# Model paths
MODEL_PATHS = {
    "Xception": "../results/xception_model.h5",
    "VGG16": "../results/vgg16_model.h5",
    "MobileNet": "../results/mobilenet_model.h5",
    "EfficientNetB0": "../results/efficientnet_model.h5"
}

# Model selection
model_choice = st.selectbox("Select a model:", list(MODEL_PATHS.keys()))
model = load_model(MODEL_PATHS[model_choice])

# Image uploader
uploaded_file = st.file_uploader(
    "Upload a rice leaf image:", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Prediction
    if st.button("Analyze Image"):
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        class_labels = ["Bacterialblight", "Blast", "Brownspot", "Tungro"]
        st.success(f"Prediction: {class_labels[predicted_class]}")

# Model comparison graph
if st.checkbox("Show model comparison"):
    if os.path.exists("results/model_scores.json"):
        with open("results/model_scores.json", "r") as file:
            scores = json.load(file)

        labels = list(scores.keys())
        acc = [scores[m]["accuracy"] for m in labels]
        f1 = [scores[m]["f1_score"] for m in labels]

        fig, ax = plt.subplots()
        x = np.arange(len(labels))
        ax.bar(x - 0.2, acc, width=0.4, label='Accuracy')
        ax.bar(x + 0.2, f1, width=0.4, label='F1-Score')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("model_scores.json not found.")
