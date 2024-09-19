import joblib
import sys

# import libraries
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# import libraries

# import relevant functions/modules from the sklearn
from sklearn.model_selection import train_test_split

from train_classifier import evaluate_model, load_data


def load_model(model_filepath):
    """
    Load a serilized model

    Args:
        model_filepath (str): The path to load the model from

    Returns:
        Classifier: The classifier object loaded from the path
    """
    # load model
    model = joblib.load(model_filepath)
    return model


def evaluate(database_filepath, model_filepath):
    """
    Utility function to evaluate a serialized classifier.
    It relies in using the same random_state to get the same 
    data split used to train the model

    Args:
        database_filepath (str): A path to the data
        model_filepath (string): A path to the serialized model
    """
    print('Loading data...\n    DATABASE: {}'.format(database_filepath, 'SELECT * FROM Tweets'))
    X, Y, category_names = load_data(database_filepath, 'SELECT * FROM Tweets')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    print('Loading model...')
    model = load_model(model_filepath)

    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)


def main():
    """
    Entry point to run the evluation script in the serialized model

    Sys Args:
        Path to the database file, it should have the cleaned data
        Path to the serialized model file
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        evaluate(database_filepath, model_filepath)

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()