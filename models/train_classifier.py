import joblib
import pandas as pd
import pickle
import re
import sys
from sqlalchemy import create_engine

# import libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import RegexpTokenizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# import libraries

# import relevant functions/modules from the sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator,TransformerMixin


def load_data(database_filepath, query):
    """
    Load a SQL Lite DB, run a query and return the results,

    Args:
        database_filepath (str): A path to the SQL Lite DB
        query (str): A query to execute against that Database

    Returns:
        DataFrame: A dataframe with only the message column (X)
        DataFrame: A dataframe with the category columns (Y)
        List[str]: A list of the category names
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql(query, engine)
    # Drop Columns where all values are 0 since those contain no information
    df = df.loc[:, (df != 0).any(axis=0)]
    X = df['message'] # , 'genre']]
    Y = df.iloc[:,4:]
    return X, Y, Y.columns


class CheckInputs(BaseEstimator, TransformerMixin):
    """
    Transformer class to print debug information in a Pipeline
    it does not modify any data.
    """

    def fit(self, X, Y):
        """
        Prints the shapes of X and Y

        Args:
            X (DataFrame): The X (features) dimension for the model
            Y (DataFrame): The Y (labels) dimension for the model

        Returns:
            self: since it is a transformer, the fit method 
                returns self
        """
        print("CheckInputs", X.shape)
        print(X)
        print("CheckInputs", Y.shape)
        return self

    def transform(self, X):
        """
        A no-op transformer

        Args:
            X (DataFrame): The X dimension for the model

        Returns:
            DataFrame: The inputs unchanged
        """
        return X


class Normalizer(BaseEstimator, TransformerMixin):
    """
    A Transformer to clean (normalize) the data further:
    
        * Convert the message to all lowercase
        * Replace the URLs with a <URL> place holder
        * Remove punctuation from the message
    """

    def fit(self, X, y=None):
        """
        No-op

        Returns:
            self: since it is a transformer, the fit method 
                returns self
        """
        return self

    def remove_url(self, text):
        # Replace all urls with a urlplaceholder string
        url_regex = 'http[s]?://(?:.)+'
        
        # Extract all the urls from the provided text 
        detected_urls = re.findall(url_regex, text)
        
        # Replace url with a url placeholder string
        for detected_url in detected_urls:
            text = text.replace(detected_url, "<URL>")
        return text

    def transform(self, X):
        """
        Transforms the data:
            * Convert the message to all lowercase
            * Replace the URLs with a <URL> place holder
            * Remove punctuation from the message
        
        Args:
            X (DataFrame): The X dimension for the model

        Returns:
            DataFrame: The transformed dataframe
        """        
        X_series = pd.Series(X) if type(X) == list else X
        # Convert to lower case
        X_series = X_series.apply(lambda t: t.lower())
        # Remove URLs
        X_series = X_series.apply(self.remove_url)
        # Remove punctuation
        X_series = X_series.apply(lambda t: re.sub(r"[^a-zA-Z0-9]", " ", t))
        
        return X_series


class StemmerTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that Stems to the inputs
    
    """
    def __init__(self):
        self.stemmer = SnowballStemmer("english", ignore_stopwords=True) 
        self.stemmer.stem("house")  
        self.stop_words = set(stopwords.words('english'))
        self.punctuation_remover = RegexpTokenizer(r'\w+')

    def run_stemmer(self, text):
        """
        Applies the stemmer to a text

        Args:
            text (str): a string to apply stemming to

        Returns:
            str: the string with the words replaced by its stem
        """
        without_stop_words = [w for w in 
                              self.punctuation_remover.tokenize(text.lower())
                              if w not in self.stop_words]
        stemmed = [self.stemmer.stem(w) for w in without_stop_words]
        return ' '.join(stemmed)
       
    def fit(self, X, y=None):
        """
        No-op

        Returns:
            self: since it is a transformer, the fit method 
                returns self
        """
        return self

    def transform(self, X):
        """
        Transforms the data:
            * Convert the message to all lowercase
            * Stemm the message
        
        Args:
            X (DataFrame): The X dimension for the model

        Returns:
            DataFrame: The transformed dataframe
        """        
        X_series = pd.Series(X) if isinstance(X, list) else X
        # Convert to lower case
        X_series = X_series.apply(lambda t: t.lower())
        # Run the stemmer
        X_series = X_series.apply(self.run_stemmer)

        return X_series


def build_model(X, Y, category_names):
    """
    Builds a model and trains it using a GridSearchCV on the input data

    Args:
        X (DataFrame): a data frame containing the features
        Y (DataFrame): a data frame containing the labels
        category_names (List[str]): a list with the label names

    Returns:
        Classifier: the best trained classifier 
    """
    score_func = classification_report_runner(
        category_names, ('weighted avg', 'f1-score'))

    parameters = {
        #'cv__max_df': (0.9, 0.95, 0.99),
        #'cv__min_df': (0.01, 0.05, 0.1),

        # Randomforest hyperparameters
        # 'clf__estimator__n_estimators': [5, 25, 50, 100],
        # "clf__estimator__max_depth":[2, 4, 8, 16],
        # "clf__estimator__ccp_alpha":[0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0],
        
        # AdaBoost Hyperparameters
        #'clf__estimator__learning_rate':[0.5, 1.0],
        #'clf__estimator__n_estimators':[10,20]
        
        # MLP Classifier
        #'mlp__alpha': 10.0 ** -np.arange(1, 5),
        # 'mlp__learning_rate': ['adaptive'], #['constant', 'invscaling', 'adaptive'],
        # 'mlp__max_iter': [250],
        # 'mlp__hidden_layer_sizes': ((70), (35, 35), (70, 35)),
                                    #  (35, 35), (70, 35)),
        # 'mlp__tol': (1e-3, 5e-3, 1e-4),
        
        # Multioutput - MLP Classifier
        #'mo_clf__estimator__alpha': 10.0 ** -np.arange(1, 5),
        'mo_clf__estimator__learning_rate': ['invscaling', 'adaptive'],
        'mo_clf__estimator__max_iter': [500],
        'mo_clf__estimator__hidden_layer_sizes': [(8), (16), (32)],
        'mo_clf__estimator__tol': (5e-2, 5e-3, 5e-4),
    }

    pipeline =  Pipeline([
        ('normalize', Normalizer()),
        ('stemmer', StemmerTransformer()),
        ('cv', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        #('cv', TfidfVectorizer(tokenizer=tokenize)),
        #('clf', MultiOutputClassifier(RandomForestClassifier()))
        #('mlp', MLPClassifier(random_state=42))
        ('mo_clf', MultiOutputClassifier(MLPClassifier(random_state=42)))
    ])

    grid = GridSearchCV(
        pipeline,
        parameters,
        cv=5,
        scoring=score_func,
        n_jobs=-1)

    print('Training model...')
    with joblib.parallel_config(backend='threading'):
        grid.fit(X, Y)
    
    print(f"Best Parameters: {grid.best_params_}")
    return grid.best_estimator_


def classification_report_runner(category_names, as_scorer_metrics=None):
    """
    A utility function to run the classification report for console reporting
    or as a scoring function in a Cross Validation run.

    Args:
        category_names (List[str]): The names of the labels
        as_scorer_metrics ((str, str)): A sequence of two strings to select
            which metric to return, if None then the complete report is
            returned as a string

    Returns:
        float | str: the value of the metric requested or 
            the complete report as a string if no metric is requested.
    """
    if as_scorer_metrics:
        avg, metric = as_scorer_metrics
        return lambda Y_test, Y_pred: classification_report(
            Y_test, Y_pred, target_names=category_names,
            output_dict=True)[avg][metric]
    else:
        return lambda Y_test, Y_pred: classification_report(
            Y_test, Y_pred, target_names=category_names)


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Utility function to run the evaluation of an existing model

    Args:
        model (Classifier): The model to evaluate
        X_test (DataFrame): The features DataFrame
        Y_test (DataFrame): The expected labels
        category_names (DataFrame): The label names
    """
    Y_pred = model.predict(X_test)
    report_runner = classification_report_runner(category_names)
    report = report_runner(Y_test, Y_pred)

    # Print the classification report
    print("Classification Report:\n", report)


def save_model(model, model_filepath):
    """
    Serialize a model to reuse it in the web application

    Args:
        model (Classifier): The model to serialize
        model_filepath (str): The pat to save the model
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    The entry point to run the training script.

    Sys Args:
        Path to the messages file.
        Path to the serilized the resulting model.
  
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath, 'SELECT * FROM Tweets'))
        X, Y, category_names = load_data(database_filepath, 'SELECT * FROM Tweets')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
        print(X_train.shape, Y_train.shape)
        print('Building model...')
        model = build_model(X_train, Y_train)
        
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