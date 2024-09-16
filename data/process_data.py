import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages,categories,on='id')

def clean_data(df):
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
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql(table, engine, index=False)  


def main():
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