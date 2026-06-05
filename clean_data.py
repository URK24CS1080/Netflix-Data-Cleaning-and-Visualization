#!/usr/bin/env python3
"""
clean_data.py

Professional data cleaning script for the `netflix_titles.csv` dataset.
Place this script in the same folder as `netflix_titles.csv` and run it
to produce `cleaned_netflix.csv` in the same folder.

This script uses only pandas and includes detailed comments for each step.
"""

import pandas as pd


def main():
    # --- 1) Load the dataset ---
    try:
        df = pd.read_csv('netflix_titles.csv')
    except FileNotFoundError:
        print("Error: 'netflix_titles.csv' not found in this folder.")
        return
    except Exception as e:
        print(f"Error loading 'netflix_titles.csv': {e}")
        return

    # --- 2) Print the number of rows before cleaning ---
    rows_before = df.shape[0]
    print(f"Rows before cleaning: {rows_before}")

    # --- 3) Display missing values for every column (initial) ---
    print('\nMissing values (initial, per column):')
    print(df.isnull().sum())

    # --- 4) Check for duplicate records and remove them if present ---
    # Count duplicates before removal
    duplicates_before = int(df.duplicated().sum())
    print(f"\nDuplicate rows found: {duplicates_before}")

    # Remove duplicate rows keeping the first occurrence
    if duplicates_before > 0:
        df = df.drop_duplicates().copy()

    # --- 5) Handle missing values for specific columns ---
    # Normalize empty strings to NaN for consistent missing detection
    df = df.replace({'': pd.NA})

    # Helper: fill missing values for a column with a specified value
    def fill_missing(col_name, fill_value):
        # Replace blank-like strings (whitespace-only) with NA first
        df[col_name] = df[col_name].replace(r'^\s*$', pd.NA, regex=True)
        df[col_name] = df[col_name].fillna(fill_value)

    # director → replace with "Unknown Director"
    if 'director' in df.columns:
        fill_missing('director', 'Unknown Director')

    # cast → replace with "Unknown Cast"
    if 'cast' in df.columns:
        fill_missing('cast', 'Unknown Cast')

    # country → replace with "Unknown Country"
    if 'country' in df.columns:
        fill_missing('country', 'Unknown Country')

    # rating → replace with the most frequent rating (mode)
    if 'rating' in df.columns:
        rating_mode = df['rating'].mode(dropna=True)
        if not rating_mode.empty:
            chosen_rating = rating_mode.iloc[0]
        else:
            # Fallback if no mode (e.g., all values missing)
            chosen_rating = 'Unknown Rating'
        print(f"\nChosen rating mode for filling missing: {chosen_rating}")
        fill_missing('rating', chosen_rating)

    # duration → replace with the most frequent duration (mode)
    if 'duration' in df.columns:
        duration_mode = df['duration'].mode(dropna=True)
        if not duration_mode.empty:
            chosen_duration = duration_mode.iloc[0]
        else:
            chosen_duration = 'Unknown Duration'
        print(f"Chosen duration mode for filling missing: {chosen_duration}")
        fill_missing('duration', chosen_duration)

    # --- 6) Remove rows where date_added is missing ---
    # Some rows may have empty strings; normalize them first then drop
    if 'date_added' in df.columns:
        df['date_added'] = df['date_added'].replace(r'^\s*$', pd.NA, regex=True)
        rows_before_drop_na_date = df.shape[0]
        df = df.dropna(subset=['date_added']).copy()
        rows_after_drop_na_date = df.shape[0]
        dropped_date_na = rows_before_drop_na_date - rows_after_drop_na_date
    else:
        dropped_date_na = 0

    # --- 7) Convert date_added into datetime format ---
    if 'date_added' in df.columns:
        # Use errors='coerce' to convert invalid formats to NaT
        df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
        # If any conversion produced NaT, drop those rows (they are invalid dates)
        invalid_dates = int(df['date_added'].isna().sum())
        if invalid_dates > 0:
            print(f"Rows with invalid `date_added` after conversion: {invalid_dates} (they will be removed)")
            df = df[df['date_added'].notna()].copy()

    # --- 8) Verify data types after cleaning ---
    print('\nData types after cleaning:')
    print(df.dtypes)

    # --- 9) Display remaining missing values ---
    print('\nRemaining missing values (per column):')
    remaining_missing = df.isnull().sum()
    print(remaining_missing)

    # --- 10) Print the number of rows after cleaning ---
    rows_after = df.shape[0]
    print(f"\nRows after cleaning: {rows_after}")

    # --- 11) Save the cleaned dataset ---
    output_filename = 'cleaned_netflix.csv'
    try:
        df.to_csv(output_filename, index=False)
        print(f"\nCleaned dataset saved to: {output_filename}")
    except Exception as e:
        print(f"Error saving cleaned dataset: {e}")

    # --- 12) Generate a short cleaning summary ---
    duplicates_removed = duplicates_before
    missing_values_handled = {
        'director': int((df['director'] == 'Unknown Director').sum()) if 'director' in df.columns else 0,
        'cast': int((df['cast'] == 'Unknown Cast').sum()) if 'cast' in df.columns else 0,
        'country': int((df['country'] == 'Unknown Country').sum()) if 'country' in df.columns else 0,
        'rating_filled_with_mode': int((df['rating'] == chosen_rating).sum()) if 'rating' in df.columns else 0,
        'duration_filled_with_mode': int((df['duration'] == chosen_duration).sum()) if 'duration' in df.columns else 0,
    }

    print('\n=== Cleaning Summary ===')
    print(f"Rows before cleaning: {rows_before}")
    print(f"Rows after cleaning: {rows_after}")
    print(f"Duplicate rows removed: {duplicates_removed}")
    print(f"Rows removed because `date_added` was missing or invalid: {dropped_date_na + (invalid_dates if 'date_added' in df.columns else 0)}")
    print('\nMissing values handled (approximate counts):')
    for k, v in missing_values_handled.items():
        print(f"- {k}: {v}")


if __name__ == '__main__':
    main()
