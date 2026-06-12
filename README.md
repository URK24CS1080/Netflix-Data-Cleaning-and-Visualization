Netflix Data Cleaning and Visualization Project

Project Overview
This project was completed as part of a Data Analytics Internship.
The objective was to clean, process, and visualize the Netflix Movies and TV Shows dataset using Python.

Dataset
Netflix Movies and TV Shows Dataset (Kaggle)

Total Records: 8807
Technologies Used
- Python
- Pandas
- Matplotlib
- Seaborn
- VS Code

Tasks Performed
Data Exploration
- Examined dataset structure
- Checked data types
- Identified missing values
- Checked duplicate records

Data Cleaning
- Handled missing values
- Converted date columns to proper format
- Removed invalid records
- Created cleaned dataset

Data Visualization
Generated the following visualizations:

1. Movies vs TV Shows Count
2. Top 10 Countries Producing Netflix Content
3. Distribution of Content Ratings
4. Release Year Distribution
5. Top 10 Genres on Netflix

Key Insights
- Movies significantly outnumber TV Shows on Netflix.
- United States contributes the highest amount of Netflix content.
- TV-MA is the most common content rating.
- Most Netflix content was released after 2010.
- International Movies and Dramas are among the most popular genres.

Project Files
- explore_data.py
- clean_data.py
- visualize_data.py
- cleaned_netflix.csv
- Visualization PNG files

# Task 2 - Predictive Modeling Using Machine Learning

## Objective

The goal of this project is to build a Machine Learning model that predicts whether a Netflix title is a Movie or a TV Show based on its features.

## Dataset

Netflix Movies and TV Shows Dataset

## Data Preparation

* Loaded the cleaned Netflix dataset.
* Selected relevant features for machine learning.
* Handled missing values.
* Encoded categorical variables into numerical format.
* Created a machine-learning-ready dataset.

## Machine Learning Algorithms Used

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

## Model Comparison

| Model                    | Accuracy |
| ------------------------ | -------- |
| Logistic Regression      | 76.12%   |
| Decision Tree Classifier | 99.89%   |
| Random Forest Classifier | 99.60%   |

## Best Model

Decision Tree Classifier

## Evaluation Metrics

* Accuracy: 99.89%
* Precision: 99.61%
* Recall: 100.00%
* F1 Score: 99.81%

## Visualizations

* Confusion Matrix
* ROC Curve

## Conclusion

The Decision Tree Classifier achieved the highest accuracy and overall performance among all tested models. The model successfully classified Netflix content as Movies or TV Shows with excellent predictive accuracy.

