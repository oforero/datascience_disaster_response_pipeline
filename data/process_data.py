import os
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads the data from two CSV files and returns it as a single 
    pandas dataframe.

    Args:
        messages_filepath (str): Path to the CSV file containing the messages
        categories_filepath (str): Path to the CSV file containing the categories 
            assigne to each message in the training dataset

    Returns:
        DataFrame: a Pandas DataFrame containing the messages with assigned categories
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages, categories, on='id')


def clean_data(df):
    """
    Converts the input data.
        * It transform the list of categories from a semicolon separated list to a colum for category
        * Extract the assignment to the category 
        * Convert any number bigger than 0 in the category assignment to 1
        * Delete the original Categories column
        * Remove duplicate rows from the dataset  

    Args:
        df (DataFrame): a DataFrame to be cleaned up, it should have a 
            categories column with a semicolon separated list (E.g. "realted-0;offer-2...")

    Returns:
        DataFrame: The cleaned DataFrame with a column per category 
            and a 0 or 1 in each row/column
    """
    splitted = df["categories"].str.split(";", expand=True)
    
    # select the first row of the categories dataframe
    row = splitted.iloc[0]
    category_colnames = row.apply(lambda s: s[:-2] if s != "id" else "id")
    splitted.columns = category_colnames
    for col in category_colnames:
        splitted[col] = splitted[col].str[-1].astype(int)
        splitted[col] = splitted[col].apply(lambda x: 1 if x > 0 else 0)
        
    splitted["id"] = df["id"]  
    df = df.join(splitted, rsuffix="_del")  
    df = df.drop(["categories", "id_del"], axis=1)
    return df.drop_duplicates()


def save_data(df, database_filename, table):
    """
    Save the dataframe to a SQL Lite DB
    

    Args:
        df (DataFrame): The cleaned input data as a DataFrame
        database_filename (str): A path to create the database file.
            If a file exists it will be deleted and a new one will be created.
        table (str): The name of the table to store the data in.
    """
    if os.path.exists(database_filename):
        os.remove(database_filename)
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql(table, engine, index=False)  


def main():
    """
    Runs the data processing script.
    
    Sys Args:
        Path to the messages file
        Path to the categories file, 
            it must have the same number of rows as the messages file
        A path that will be use to create the database and store the
            cleaned data
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        print(df.head())

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath, 'Tweets')
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()