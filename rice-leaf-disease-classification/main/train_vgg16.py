import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.model_vgg16 import build_vgg16_model
from utils.preprocessing import load_data

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def main():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data")
    train_gen, val_gen = load_data(data_path, image_size=(224, 224), batch_size=32)

    model = build_vgg16_model(input_shape=(224, 224, 3), num_classes=4)

    checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "results", "vgg16_model.h5")
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    earlystop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=2,
        callbacks=[checkpoint, earlystop]
    )

if __name__ == "__main__":
    main()
