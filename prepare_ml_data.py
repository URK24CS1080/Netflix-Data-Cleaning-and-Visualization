#!/usr/bin/env python3
"""
prepare_ml_data.py

Prepare the cleaned Netflix dataset for a machine learning task.

Target variable:
- type

This script:
- loads cleaned_netflix.csv from the parent project folder
- keeps only the required columns
- fills missing values with "Unknown"
- encodes categorical features with LabelEncoder
- prints encoding mappings
- saves the ML-ready dataset as task2_ml/netflix_ml_ready.csv
"""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def main():
	# Use the folder containing this script as the base for all file paths.
	script_dir = Path(__file__).resolve().parent

	# The cleaned dataset is stored in the parent project folder.
	input_file = script_dir.parent / "cleaned_netflix.csv"

	# The processed ML-ready dataset should be saved inside task2_ml.
	output_file = script_dir / "netflix_ml_ready.csv"

	# Load the cleaned Netflix dataset.
	try:
		df = pd.read_csv(input_file)
	except FileNotFoundError:
		print(f"Error: '{input_file.name}' was not found in the parent folder.")
		return
	except Exception as error:
		print(f"Error loading '{input_file}': {error}")
		return

	# Keep only the columns needed for this ML task.
	selected_columns = ["type", "release_year", "rating", "country", "listed_in"]
	df = df[selected_columns].copy()

	# Print the dataset shape after selecting the required features.
	print("Dataset shape:", df.shape)

	# Check missing values before cleaning them.
	print("\nMissing value summary before filling:")
	print(df.isnull().sum())

	# Replace empty strings with missing values so they are handled consistently.
	df = df.replace("", pd.NA)

	# Fill any missing values with the label "Unknown" as requested.
	df = df.fillna("Unknown")

	# Verify missing values after filling.
	print("\nMissing value summary after filling:")
	remaining_missing = df.isnull().sum()
	print(remaining_missing)

	# Prepare label encoders for each categorical column.
	categorical_columns = ["type", "rating", "country", "listed_in"]
	encoders = {}

	# Encode each categorical column and store the mapping for later display.
	for column in categorical_columns:
		encoder = LabelEncoder()
		df[column] = encoder.fit_transform(df[column].astype(str))
		encoders[column] = encoder

	# Print all encoding mappings so the preprocessing is transparent.
	print("\nEncoding mappings:")
	for column, encoder in encoders.items():
		mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
		print(f"\n{column} mapping:")
		print(mapping)

	# Final check to make sure no missing values remain.
	final_missing_total = int(df.isnull().sum().sum())
	print("\nRemaining missing values total:", final_missing_total)

	# Save the ML-ready dataset inside the task2_ml folder.
	df.to_csv(output_file, index=False)

	# Print the number of rows processed for a quick summary.
	print("\nNumber of rows processed:", df.shape[0])
	print("Saved ML-ready dataset to:", output_file)


if __name__ == "__main__":
	# Run the preprocessing workflow when the script is executed directly.
	main()

