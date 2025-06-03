import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tensorflow.keras.models import load_model
from utils.preprocessing import load_data
from utils.evaluation import evaluate_with_metrics

def main():
    model_path = os.path.join(os.path.dirname(__file__), "..", "results", "vgg16_model.h5")
    model = load_model(model_path)

    data_path = os.path.join(os.path.dirname(__file__), "..", "data")
    _, val_gen = load_data(data_path, image_size=(224, 224), batch_size=32)

    loss, accuracy = model.evaluate(val_gen)
    print("✅ Evaluation Complete")
    print(f"🔹 Validation Loss: {loss:.4f}")
    print(f"🔹 Validation Accuracy: {accuracy:.4f}")

    report, cm = evaluate_with_metrics(model, val_gen)
    print("\n📊 Classification Report:\n", report)
    print("\n📉 Confusion Matrix:\n", cm)

if __name__ == "__main__":
    main()
