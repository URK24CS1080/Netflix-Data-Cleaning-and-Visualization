#!/usr/bin/env python3
"""
visualize_data.py

Data visualization script for the cleaned Netflix dataset.
The file `cleaned_netflix.csv` must be in the same folder as this script.

This script creates professional-looking charts, saves them as PNG files,
and displays each chart on screen using matplotlib.
"""

import os
import warnings

import pandas as pd
import seaborn as sns
import matplotlib

# Use a non-interactive backend so the script runs reliably in terminal sessions.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_visualizations_folder(folder_name="visualizations"):
    # Create the output folder only if it does not already exist.
    os.makedirs(folder_name, exist_ok=True)


def save_and_show_plot(output_path):
    # Save the current figure as a PNG file, then display it, and finally close it.
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    # Matplotlib may use a non-interactive backend in terminal environments.
    # We still call plt.show() as requested, while suppressing the backend warning.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive*")
        plt.show()
    plt.close()


def main():
    # Apply a clean seaborn style for all charts in this script.
    sns.set_theme(style="whitegrid")

    # Load the cleaned Netflix dataset from the same directory as this script.
    try:
        df = pd.read_csv("cleaned_netflix.csv")
    except FileNotFoundError:
        print("Error: 'cleaned_netflix.csv' not found in this folder.")
        return
    except Exception as error:
        print(f"Error loading 'cleaned_netflix.csv': {error}")
        return

    # Create the folder where all images will be saved.
    ensure_visualizations_folder("visualizations")

    # ----------------------------------------------------------------------
    # A. Movies vs TV Shows Count
    # ----------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    content_counts = df["type"].value_counts()
    sns.barplot(x=content_counts.index, y=content_counts.values, color="#4C72B0")
    plt.title("Movies vs TV Shows on Netflix", fontsize=16, fontweight="bold")
    plt.xlabel("Content Type", fontsize=12)
    plt.ylabel("Number of Titles", fontsize=12)
    save_and_show_plot("visualizations/movies_vs_tvshows.png")
    print("Saved: visualizations/movies_vs_tvshows.png")

    # ----------------------------------------------------------------------
    # B. Top 10 Countries Producing Netflix Content
    # ----------------------------------------------------------------------
    # Some rows can contain multiple countries separated by commas.
    # We use only the first country listed for each title.
    country_series = (
        df["country"].dropna().astype(str).str.split(",").str[0].str.strip()
    )
    top_countries = country_series.value_counts().head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_countries.values, y=top_countries.index, color="#55A868")
    plt.title("Top 10 Countries Producing Netflix Content", fontsize=16, fontweight="bold")
    plt.xlabel("Number of Titles", fontsize=12)
    plt.ylabel("Country", fontsize=12)
    save_and_show_plot("visualizations/top_countries.png")
    print("Saved: visualizations/top_countries.png")

    # ----------------------------------------------------------------------
    # C. Distribution of Content Ratings
    # ----------------------------------------------------------------------
    rating_counts = df["rating"].value_counts()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=rating_counts.index, y=rating_counts.values, color="#DD8452")
    plt.title("Distribution of Content Ratings", fontsize=16, fontweight="bold")
    plt.xlabel("Rating", fontsize=12)
    plt.ylabel("Number of Titles", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    save_and_show_plot("visualizations/content_ratings.png")
    print("Saved: visualizations/content_ratings.png")

    # ----------------------------------------------------------------------
    # D. Release Year Distribution
    # ----------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.histplot(df["release_year"], bins=20, kde=False, color="#4C72B0")
    plt.title("Release Year Distribution of Netflix Titles", fontsize=16, fontweight="bold")
    plt.xlabel("Release Year", fontsize=12)
    plt.ylabel("Number of Titles", fontsize=12)
    save_and_show_plot("visualizations/release_year_distribution.png")
    print("Saved: visualizations/release_year_distribution.png")

    # ----------------------------------------------------------------------
    # E. Top 10 Genres on Netflix
    # ----------------------------------------------------------------------
    # The listed_in column can contain multiple genres separated by commas.
    # Split them, clean whitespace, and count each genre.
    genres = (
        df["listed_in"].dropna().astype(str).str.split(",").explode().str.strip()
    )
    top_genres = genres.value_counts().head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_genres.values, y=top_genres.index, color="#8172B3")
    plt.title("Top 10 Genres on Netflix", fontsize=16, fontweight="bold")
    plt.xlabel("Number of Titles", fontsize=12)
    plt.ylabel("Genre", fontsize=12)
    save_and_show_plot("visualizations/top_genres.png")
    print("Saved: visualizations/top_genres.png")

    # Final confirmation message requested by the user.
    print("All visualizations generated successfully.")


if __name__ == "__main__":
    main()
