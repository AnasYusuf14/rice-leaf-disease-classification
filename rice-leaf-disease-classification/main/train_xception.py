import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.preprocessing import load_data
from models.model_xception import build_xception_model

def main():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data")
    image_size = (224, 224)
    batch_size = 32
    epochs = 2

    train_gen, val_gen = load_data(data_path, image_size=image_size, batch_size=batch_size)

    model = build_xception_model(input_shape=(224, 224, 3), num_classes=4)

    model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen
    )

    model.save(os.path.join(os.path.dirname(__file__), "..", "results", "xception_model.h5"))
    print("✅ Model training complete and saved to results/xception_model.h5")

if __name__ == "__main__":
    main()
