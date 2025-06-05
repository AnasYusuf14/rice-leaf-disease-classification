import json
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to trained models
model_paths = {
    "Xception": "C:/Users/wwwan/OneDrive/Desktop/rice_leaf_project/rice-leaf-disease-classification/results/xception_model.h5",
    "VGG16": "C:/Users/wwwan/OneDrive/Desktop/rice_leaf_project/rice-leaf-disease-classification/results/vgg16_model.h5",
    "MobileNet": "C:/Users/wwwan/OneDrive/Desktop/rice_leaf_project/rice-leaf-disease-classification/results/mobilenet_model.h5",
    "EfficientNetB0": "C:/Users/wwwan/OneDrive/Desktop/rice_leaf_project/rice-leaf-disease-classification/results/efficientnet_model.h5"
}

# Dataset settings
data_dir = "C:/Users/wwwan/OneDrive/Desktop/rice_leaf_project/rice-leaf-disease-classification/data"
img_size = (224, 224)
batch_size = 32
class_labels = ["Bacterialblight", "Blast", "Brownspot", "Tungro"]

# Validation data generator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Create confusion matrices for each model
results = {}

for model_name, model_path in model_paths.items():
    print(f"Processing {model_name}...")
    model = load_model(model_path)
    predictions = model.predict(val_gen, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_gen.classes
    cm = confusion_matrix(true_classes, predicted_classes)
    results[model_name] = cm.tolist()

import os
os.makedirs("results", exist_ok=True)  # Add this before opening the file

with open("C:/Users/wwwan/OneDrive/Desktop/rice_leaf_project/rice-leaf-disease-classification/results/confusion_matrices.json", "w") as f:
    json.dump(results, f, indent=2)

print("✅ confusion_matrices.json created successfully.")
