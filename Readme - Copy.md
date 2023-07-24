This project aims to perform sentiment analysis on news data using machine learning models. It uses Python and various libraries for data processing, feature extraction, and model training.

Table of Contents
Introduction
Setup
Data
Preprocessing
Feature Engineering
Machine Learning Models
Performance Evaluation
Usage
License
Introduction
Sentiment analysis is the process of determining the sentiment or emotion conveyed in a piece of text, such as news articles. This project aims to predict the sentiment for both the title and headline of news articles using machine learning models.

Setup
Before running the code, ensure you have the required libraries installed. You can use the following command to install the dependencies:

Copy code
Data
The project uses two CSV files: training_data.csv and the user-uploaded CSV file containing test data. The training data contains features like title, headline, source, and topic, along with their corresponding sentiment scores.

Preprocessing
The data preprocessing steps include handling missing values, text cleaning, and tokenization. The text is converted to lowercase, stop words are removed, and special characters are cleaned.

Feature Engineering
Various features are engineered from the text data, such as term frequency-inverse document frequency (TF-IDF) vectors and sentiment scores using the TextBlob library.

Machine Learning Models
The project uses XGBoost regressors for predicting sentiment scores for both the title and headline of news articles.

Performance Evaluation
The performance of the machine learning models is evaluated using the root mean squared error (RMSE) metric.

Usage
To run the project, execute the main script or Jupyter notebook containing the code. Ensure the required CSV files are present in the appropriate locations.
