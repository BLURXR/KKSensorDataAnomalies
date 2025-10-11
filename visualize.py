import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

df = pd.read_csv("supervised_predictions.csv")

y_true = df["actual_fail"]
y_pred = df["predicted_fail"]

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

accuracy = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

plt.figure(figsize=(10,6))
colors = df["predicted_fail"].map({0: "blue", 1: "red"})
plt.scatter(df["CS"], df["Temperature"], c=colors, alpha=0.6, edgecolors='k')
plt.xlabel("CS Sensor")
plt.ylabel("Temperature Sensor")
plt.title(f"Sensor Data with Predicted Failures\nAccuracy: {accuracy:.2f}, Recall: {recall:.2f}")
plt.grid(True)
plt.show()

labels = ["True Positive", "True Negative", "False Positive", "False Negative"]
values = [tp, tn, fp, fn]

plt.figure(figsize=(8,6))
plt.bar(labels, values, color=["green", "green", "red", "red"])
plt.ylabel("Count")
plt.title("Prediction Outcomes")
plt.xticks(rotation=20, ha="right")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()