import json
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Use relative paths for portability
model_paths = {
    "Xception": "results/xception_model.h5",
    "VGG16": "results/vgg16_model.h5",
    "MobileNet": "results/mobilenet_model.h5",
    "EfficientNetB0": "results/efficientnet_model.h5"
}

# Dataset settings (relative path)
data_dir = "data"
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

# Ensure results directory exists and save the confusion matrices
os.makedirs("results", exist_ok=True)
with open("results/confusion_matrices.json", "w") as f:
    json.dump(results, f, indent=2)

print("✅ confusion_matrices.json created successfully.")