import sys
import pandas as pd
from sqlalchemy import create_engine
import sqlite3


def load_data(messages_filepath, categories_filepath):
    """
    read messages and categories files into a pandas dataframe.
    Merge the two data frames.
    :param messages_filepath:
    :param categories_filepath:
    :return: df  pandas dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id', how='inner')
    return df


def clean_data(df):
    """
    clean dataframe and return it.
    :param df: rpandas dataframe
    :return: df pandas dataframe

    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[1, :]
    category_names = row.apply(lambda x: x[:-2])
    categories.columns = category_names
    for column in categories:
        categories[column] = categories[column].apply(lambda x: int(x[-1]))
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    """
    save pandas dataframe to sqlite Database.
    :param df:
    :param database_filename:
    :return: None
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Response', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
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
