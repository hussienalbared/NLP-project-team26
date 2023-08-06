import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def plot_confusion_matrix(test_labels, predictions):
    # plot Confusion Matrix

    cm = pd.DataFrame(
        confusion_matrix(test_labels, predictions),
        index=["Not Sarcastic", "Sarcastic"],
        columns=["Not Sarcastic", "Sarcastic"],
    )

    fig = plt.figure(figsize=(6, 4))
    ax = sns.heatmap(
        cm, annot=True, cbar=False, cmap="Blues", linewidths=0.5, fmt=".0f"
    )
    ax.set_title("SARCASM DETECTION CONFUSION MATRIX", fontsize=16, y=1.25)
    ax.set_ylabel("Actual", fontsize=14)
    ax.set_xlabel("Predicted", fontsize=14)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(labelsize=12)


def evaluate_model(model, test_features, test_labels):
    predictions = model.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    report = classification_report(test_labels, predictions)
    print("Accuracy: ", accuracy)
    print("F1 Score: ", f1)
    print("Classification Report: \n", report)
    plot_confusion_matrix(test_labels, predictions)
