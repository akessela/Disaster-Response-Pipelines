import sys
import pandas as pd
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier

import pickle as pk


def load_data(database_filepath):
    """
    read data from a database into pandas dataframe. Split data into predictor (X) and
    response(y)
    :param database_filepath: filepath for source database
    :return:
        X - pandas dataframe (predictor)
        y - pandas dataframe ( response)
    """
    engine = create_engine('sqlite:///{}.db'.format(database_filepath))
    df = pd.read_sql('SELECT * FROM Response', engine)
    x = df['message']
    y = df.iloc[:, 4:]
    return x, y


def tokenize(text):
    """
    Tokenize text by removing stop words, lemmatization, normalization
    :param text: str or unicode input text to tokenize
    :return: tokens : list of tokenized words
    """
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens


def build_model():
    """
    build model using GridSearchCV
    :return: cv - GridSearchCV object
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [20, 40],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, x_test, y_test, category_names):
    """
    Predict on test data. Report f1 score, preiction and recall for each output category of the dataset.
    response has multiple output classes
    :param model: trained model
    :param x_test:pandas dataframe. predictor test data for model testing.
    :param y_test:pandas dataframe. true labels for test data
    :param category_names: category names for y_test
    :return:None
    """
    y_pred = model.predict(x_test)
    for i, column in enumerate(y_test.columns):
        print(classification_report(y_test[column], y_pred[:, i]))


def save_model(model, model_filepath):
    pk.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()