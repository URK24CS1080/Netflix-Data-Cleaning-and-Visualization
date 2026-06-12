#!/usr/bin/env python3
"""
evaluate_model.py

Evaluate the saved machine learning model for the Netflix classification task.

This script:
- loads task2_ml/netflix_ml_ready.csv
- loads task2_ml/models/best_model.pkl
- splits the data for evaluation
- calculates common classification metrics
- generates professional evaluation visualizations
- saves plots to task2_ml/ml_visualizations
"""

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
	roc_curve,
)
from sklearn.model_selection import train_test_split


def ensure_folder(folder_path: Path):
	# Create the visualization folder if it does not already exist.
	folder_path.mkdir(parents=True, exist_ok=True)


def get_score_values(model, X_test):
	# Use probability scores when available so ROC curves are meaningful.
	if hasattr(model, "predict_proba"):
		return model.predict_proba(X_test)[:, 1]

	# Fall back to decision scores if the model does not support probabilities.
	if hasattr(model, "decision_function"):
		scores = model.decision_function(X_test)
		return scores

	# If neither is available, ROC analysis cannot be computed reliably.
	return None


def main():
	# Use the current script location so the project works regardless of where it is launched from.
	script_dir = Path(__file__).resolve().parent

	# Define the paths to the data, model, and output folders.
	data_file = script_dir / "netflix_ml_ready.csv"
	model_file = script_dir / "models" / "best_model.pkl"
	viz_dir = script_dir / "ml_visualizations"

	# Create the visualization folder if needed.
	ensure_folder(viz_dir)

	# Load the ML-ready dataset.
	try:
		df = pd.read_csv(data_file)
	except FileNotFoundError:
		print(f"Error: '{data_file.name}' not found in {script_dir}.")
		return
	except Exception as error:
		print(f"Error loading '{data_file}': {error}")
		return

	# Load the previously saved best model.
	try:
		with open(model_file, "rb") as file:
			model = pickle.load(file)
	except FileNotFoundError:
		print(f"Error: '{model_file.name}' not found in {model_file.parent}.")
		return
	except Exception as error:
		print(f"Error loading model from '{model_file}': {error}")
		return

	# Separate target and features.
	# `type` is the target column, and all other columns are features.
	X = df.drop(columns=["type"])
	y = df["type"]

	# Split the dataset for evaluation.
	# An 80/20 split with random_state=42 keeps the result reproducible.
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.2,
		random_state=42,
		stratify=y,
	)

	# Generate predictions for the test set.
	y_pred = model.predict(X_test)

	# Calculate standard classification metrics.
	accuracy = accuracy_score(y_test, y_pred)
	precision = precision_score(y_test, y_pred, average="binary", zero_division=0)
	recall = recall_score(y_test, y_pred, average="binary", zero_division=0)
	f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)

	# Print the core evaluation metrics.
	print(f"Accuracy: {accuracy:.4f}")
	print(f"Precision: {precision:.4f}")
	print(f"Recall: {recall:.4f}")
	print(f"F1 Score: {f1:.4f}")

	# Show the full classification report for both classes.
	print("\nClassification Report:")
	print(classification_report(y_test, y_pred, zero_division=0))

	# ------------------------------------------------------------------
	# A. Confusion Matrix Heatmap
	# ------------------------------------------------------------------
	sns.set_theme(style="whitegrid")
	cm = confusion_matrix(y_test, y_pred)
	plt.figure(figsize=(8, 6))
	sns.heatmap(
		cm,
		annot=True,
		fmt="d",
		cmap="Blues",
		cbar=False,
		xticklabels=["Predicted 0", "Predicted 1"],
		yticklabels=["Actual 0", "Actual 1"],
	)
	plt.title("Confusion Matrix", fontsize=16, fontweight="bold")
	plt.xlabel("Predicted Label", fontsize=12)
	plt.ylabel("Actual Label", fontsize=12)
	plt.tight_layout()
	confusion_matrix_path = viz_dir / "confusion_matrix.png"
	plt.savefig(confusion_matrix_path, dpi=300, bbox_inches="tight")
	plt.show()
	plt.close()
	print(f"Saved: {confusion_matrix_path}")

	# ------------------------------------------------------------------
	# B. ROC Curve
	# ------------------------------------------------------------------
	score_values = get_score_values(model, X_test)
	if score_values is not None:
		fpr, tpr, _ = roc_curve(y_test, score_values)
		roc_auc = roc_auc_score(y_test, score_values)

		plt.figure(figsize=(8, 6))
		plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})", color="#2C7FB8", linewidth=2)
		plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
		plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=16, fontweight="bold")
		plt.xlabel("False Positive Rate", fontsize=12)
		plt.ylabel("True Positive Rate", fontsize=12)
		plt.legend(loc="lower right")
		plt.tight_layout()
		roc_curve_path = viz_dir / "roc_curve.png"
		plt.savefig(roc_curve_path, dpi=300, bbox_inches="tight")
		plt.show()
		plt.close()
		print(f"Saved: {roc_curve_path}")
	else:
		print("ROC curve could not be generated because the model does not provide score values.")

	# ------------------------------------------------------------------
	# C. Feature Importance Plot for Random Forest models only
	# ------------------------------------------------------------------
	if model.__class__.__name__ == "RandomForestClassifier" and hasattr(model, "feature_importances_"):
		importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)

		plt.figure(figsize=(10, 6))
		sns.barplot(x=importances.values, y=importances.index, color="#55A868")
		plt.title("Top Feature Importances", fontsize=16, fontweight="bold")
		plt.xlabel("Importance Score", fontsize=12)
		plt.ylabel("Feature", fontsize=12)
		plt.tight_layout()
		feature_importance_path = viz_dir / "feature_importance.png"
		plt.savefig(feature_importance_path, dpi=300, bbox_inches="tight")
		plt.show()
		plt.close()
		print(f"Saved: {feature_importance_path}")

	# Print a concise summary of the evaluation results.
	print("\nEvaluation Summary:")
	print(f"Accuracy: {accuracy:.4f}")
	print(f"Precision: {precision:.4f}")
	print(f"Recall: {recall:.4f}")
	print(f"F1 Score: {f1:.4f}")


if __name__ == "__main__":
	# Run the evaluation workflow when the script is executed directly.
	main()

