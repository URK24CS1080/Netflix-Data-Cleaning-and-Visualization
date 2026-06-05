#!/usr/bin/env python3
"""
create_visualizations_folder.py

Beginner-friendly utility script that checks whether a folder named
`visualizations` exists in the current project folder and creates it if
needed.

This script uses only Python's standard library (`pathlib`).
"""

from pathlib import Path


def main():
    # Define the folder name we want to use for saved charts and plots.
    folder_name = "visualizations"

    # Use the directory where this script is located as the project folder.
    # This keeps the script portable and easy to run from the same project.
    project_dir = Path(__file__).resolve().parent

    # Build the full path to the target folder.
    visualizations_dir = project_dir / folder_name

    # Check whether the folder already exists.
    if visualizations_dir.exists():
        # If it exists, let the user know no action was needed.
        print("visualizations folder already exists")
    else:
        # If it does not exist, create it.
        visualizations_dir.mkdir()

        # Confirm successful creation with the exact message requested.
        print("visualizations folder created successfully")


if __name__ == "__main__":
    # Run the main function when the script is executed directly.
    main()
