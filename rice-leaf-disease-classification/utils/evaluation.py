from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_with_metrics(model, val_gen):
    predictions = model.predict(val_gen)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_gen.classes
    class_labels = list(val_gen.class_indices.keys())

    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    cm = confusion_matrix(true_classes, predicted_classes)

    return report, cm
