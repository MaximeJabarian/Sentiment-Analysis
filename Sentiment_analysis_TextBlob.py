# Import necessary libraries
import pandas as pd
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data.columns = ["Tweet ID", "entity", "sentiment", "tweet_content"]
    data['tweet_content'] = data['tweet_content'].astype(str)
    data['tweet_content'] = data['tweet_content'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    data['tweet_content'] = data['tweet_content'].str.replace('[^\w\s]','')
    data['tweet_content'] = data['tweet_content'].str.replace('\d+', '')

    stop = stopwords.words('english')
    data['tweet_content'] = data['tweet_content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    st = PorterStemmer()
    data['tweet_content'] = data['tweet_content'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

    return data

# Feature extraction
def extract_features(data, vectorizer=None):
    X = data['tweet_content']
    if vectorizer is None:
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(X)
    else:
        X = vectorizer.transform(X)
    y = data['sentiment']
    return X, y, vectorizer

# Train model
def train_model(X_train, y_train):
    clf = MultinomialNB().fit(X_train, y_train)
    return clf

# Evaluate model
def evaluate_model(y_true, y_pred):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average='weighted'))
    print("Recall:", recall_score(y_true, y_pred, average='weighted'))
    print("F1-score:", f1_score(y_true, y_pred, average='weighted'))

# Predict sentiment on new data
def predict_sentiment(model, new_data, vectorizer):
    new_data = new_data.apply(lambda x: " ".join(x.lower() for x in x.split()))
    new_data = new_data.str.replace('[^\w\s]', '')
    new_data = new_data.str.replace('\d+', '')

    stop = stopwords.words('english')
    new_data = new_data.apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    st = PorterStemmer()
    new_data = new_data.apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

    new_data = vectorizer.transform(new_data)
    new_y_pred = model.predict(new_data)

    return new_y_pred

# Main
if __name__ == "__main__":
    training_file_path = "twitter_training.csv"
    validation_file_path = "twitter_validation.csv"

    # Load and preprocess training data
    data_train = load_and_preprocess_data(training_file_path)
    X_train, y_train, vectorizer = extract_features(data_train)

    # Train model
    clf = train_model(X_train, y_train)

    # Load and preprocess validation data
    data_test = load_and_preprocess_data(validation_file_path)
    X_test, y_test, _ = extract_features(data_test, vectorizer)

    # Evaluate model on validation data
    y_pred = clf.predict(X_test)
    print("Validation metrics:")
    evaluate_model(y_test, y_pred)

    # Predict sentiment on new data
    new_data = data_test["tweet_content"]
    new_y_pred = predict_sentiment(clf, new_data, vectorizer)
    data_test["sentiment predicted"] = new_y_pred

    # Evaluate model on new data
    print("New data metrics:")
    evaluate_model(y_test, new_y_pred)
