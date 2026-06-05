#!/usr/bin/env python3
"""
explore_data.py

Beginner-friendly script to perform initial exploration of the
`netflix_titles.csv` dataset. The CSV file must be in the same folder
as this script.

Requirements implemented:
- Import pandas
- Load `netflix_titles.csv`
- Display: shape, columns, head, tail, dtypes, missing counts, duplicates
- Print a short summary

Run: `python explore_data.py` (from the folder containing the CSV)
"""

import pandas as pd


def main():
    # 1) Load the dataset into a pandas DataFrame
    # We use a try/except to give a helpful message if the file is missing.
    try:
        # Since the CSV is in the same folder as this script, we can use
        # the filename directly. If you move the script, update the path.
        df = pd.read_csv('netflix_titles.csv')
    except FileNotFoundError:
        print("Error: 'netflix_titles.csv' not found in this folder.")
        return
    except Exception as e:
        print(f"Error loading 'netflix_titles.csv': {e}")
        return

    # 2) Display dataset shape (rows, columns)
    # df.shape returns a tuple: (number_of_rows, number_of_columns)
    print("\n=== Dataset Shape ===")
    print("Rows, Columns:", df.shape)

    # 3) Display column names
    # df.columns is an Index object; convert to list for a cleaner print
    print("\n=== Column Names ===")
    print(list(df.columns))

    # 4) Display the first 5 rows (useful to quickly inspect the data)
    print("\n=== First 5 Rows ===")
    print(df.head(5).to_string(index=False))

    # 5) Display the last 5 rows
    print("\n=== Last 5 Rows ===")
    print(df.tail(5).to_string(index=False))

    # 6) Show data types for each column
    # This helps identify numeric vs. object (string) columns, datetimes, etc.
    print("\n=== Data Types ===")
    print(df.dtypes)

    # 7) Count missing values in each column
    # df.isnull() returns a DataFrame of booleans; sum() counts True values per column
    print("\n=== Missing Values (per column) ===")
    missing_per_col = df.isnull().sum()
    print(missing_per_col)

    # 8) Count duplicate records
    # df.duplicated() marks rows that are duplicates of previous rows
    duplicate_count = int(df.duplicated().sum())
    print("\n=== Duplicate Records ===")
    print("Number of duplicate rows:", duplicate_count)

    # 9) Short summary of the dataset
    # Provide a concise human-readable summary using values we computed above
    total_missing = int(missing_per_col.sum())
    rows, cols = df.shape

    print("\n=== Short Summary ===")
    print(f"Total rows: {rows}")
    print(f"Total columns: {cols}")
    print(f"Total missing values (all columns): {total_missing}")
    print(f"Total duplicate rows: {duplicate_count}")

    # Non-null counts per column (useful to see which columns have lots of missing data)
    print("\nNon-null counts per column:")
    print(df.count())


if __name__ == "__main__":
    # When run as a script, call main()
    main()
