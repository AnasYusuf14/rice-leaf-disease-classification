import os
import sys

# Add project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.preprocessing import load_data

def main():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data")
    image_size = (224, 224)
    batch_size = 32

    train_gen, val_gen = load_data(data_path, image_size=image_size, batch_size=batch_size)

    print("✅ Data loaded successfully")
    print(f"🔹 Training samples: {train_gen.samples}")
    print(f"🔹 Validation samples: {val_gen.samples}")
    print(f"🔹 Class labels: {train_gen.class_indices}")

if __name__ == "__main__":
    main()
