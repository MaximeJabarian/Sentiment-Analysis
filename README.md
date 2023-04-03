# Sentiment Analysis on Twitter Data
This repository contains Python code to perform sentiment analysis on Twitter data using the Naïve Bayes classifier. The code demonstrates how to load, preprocess, and analyze the text data, train the model, and evaluate its performance.

## Dependencies

- pandas
- textblob
- scikit-learn
- nltk

You can install the required packages using pip:

pip install pandas textblob scikit-learn nltk

## Dataset
The dataset should be in CSV format and should contain the following columns:
- sentiment
- tweet_content

Two separate CSV files are used in this code:
- twitter_training.csv: The training dataset used to train the model.
- twitter_validation.csv: The validation dataset used to evaluate the model.

## Code Structure
The code is structured in the following way:
- load_and_preprocess_data(file_path): Loads and preprocesses the data, including lowercasing, removing punctuation and special characters, removing numbers, and stemming.
- extract_features(data, vectorizer=None): Extracts features from the preprocessed data using the TfidfVectorizer.
- train_model(X_train, y_train): Trains the Naïve Bayes classifier using the extracted features and sentiment labels.
- evaluate_model(y_true, y_pred): Evaluates the model using accuracy, precision, recall, and F1-score.
- predict_sentiment(model, new_data, vectorizer): Predicts the sentiment for new data using the trained model and vectorizer.

## Usage
To use the code, you can run the script sentiment_analysis.py:

python sentiment_analysis.py

This script will:
- Load and preprocess the training dataset.
- Extract features and train the model.
- Load and preprocess the validation dataset.
- Evaluate the model on the validation dataset.
- Predict the sentiment for new data and evaluate the model performance.
