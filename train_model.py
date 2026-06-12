#!/usr/bin/env python3
"""
train_model.py

Train multiple classification models on the ML-ready Netflix dataset and
save the best-performing model.

Goal:
- Predict whether Netflix content is a Movie or TV Show.

This script:
- loads task2_ml/netflix_ml_ready.csv
- separates the target column from the features
- splits the data into training and testing sets
- trains Logistic Regression, Decision Tree, and Random Forest models
- compares accuracy scores
- saves the best model to task2_ml/models/best_model.pkl
"""

from pathlib import Path
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main():
	# Use the folder containing this script as the base for relative paths.
	script_dir = Path(__file__).resolve().parent

	# Build the path to the ML-ready dataset created by the preprocessing step.
	data_file = script_dir / "netflix_ml_ready.csv"

	# Make sure the model output folder exists before saving anything.
	models_dir = script_dir / "models"
	models_dir.mkdir(parents=True, exist_ok=True)

	# Load the processed dataset.
	try:
		df = pd.read_csv(data_file)
	except FileNotFoundError:
		print(f"Error: '{data_file.name}' was not found in {script_dir}.")
		return
	except Exception as error:
		print(f"Error loading '{data_file}': {error}")
		return

	# Separate the target column from the feature columns.
	# The target is `type`, and all remaining columns are used as features.
	X = df.drop(columns=["type"])
	y = df["type"]

	# Split the dataset into training and testing sets.
	# We use an 80/20 split and random_state=42 for reproducibility.
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.2,
		random_state=42,
		stratify=y,
	)

	# Define the three models required for the internship project.
	models = {
		"Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
		"Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
		"Random Forest Classifier": RandomForestClassifier(random_state=42),
	}

	# Store each model's accuracy so we can compare them in a table.
	results = []
	trained_models = {}

	# Train and evaluate each model on the test set.
	for model_name, model in models.items():
		model.fit(X_train, y_train)
		predictions = model.predict(X_test)
		accuracy = accuracy_score(y_test, predictions)

		# Save the result for later comparison.
		results.append({"Model": model_name, "Accuracy": accuracy})
		trained_models[model_name] = model

		# Print each model's accuracy clearly for the user.
		print(f"{model_name} Accuracy: {accuracy:.4f}")

	# Create a comparison table sorted from best to worst accuracy.
	results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

	print("\nModel Comparison Table:")
	print(results_df.to_string(index=False))

	# Determine the best-performing model from the comparison table.
	best_row = results_df.iloc[0]
	best_model_name = best_row["Model"]
	best_accuracy = best_row["Accuracy"]
	best_model = trained_models[best_model_name]

	# Save the best model to the requested path using the pickle standard library.
	best_model_path = models_dir / "best_model.pkl"
	with open(best_model_path, "wb") as file:
		pickle.dump(best_model, file)

	# Print the best model details.
	print(f"\nBest Model: {best_model_name}")
	print(f"Best Accuracy: {best_accuracy:.4f}")
	print(f"Best model saved to: {best_model_path}")


if __name__ == "__main__":
	# Run the training workflow when the script is executed directly.
	main()

