import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tensorflow.keras.models import load_model
from utils.preprocessing import load_data
from utils.evaluation import evaluate_with_metrics

def main():
    model_path = os.path.join(os.path.dirname(__file__), "..", "results", "xception_model.h5")
    model = load_model(model_path)

    data_path = os.path.join(os.path.dirname(__file__), "..", "data")
    _, val_gen = load_data(data_path, image_size=(224, 224), batch_size=32)

    # Accuracy & loss
    loss, accuracy = model.evaluate(val_gen)
    print("✅ Model Evaluation Complete")
    print(f"🔹 Validation Loss: {loss:.4f}")
    print(f"🔹 Validation Accuracy: {accuracy:.4f}")

    # Advanced Metrics
    report, cm = evaluate_with_metrics(model, val_gen)
    print("\n📊 Classification Report:\n")
    print(report)
    print("\n📉 Confusion Matrix:\n")
    print(cm)

if __name__ == "__main__":
    main()
