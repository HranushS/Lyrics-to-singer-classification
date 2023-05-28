#!/usr/bin/env python
# coding: utf-8


import re
import pandas as pd
import numpy as np
import random
import nltk
from textblob import TextBlob
from datetime import datetime
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def number_of_lines(df, column_name):
    """
    Counts the number of lines in a specified column of a DataFrame and adds a new 'lines' column to the DataFrame.

    Args:
        df (DataFrame): The DataFrame to process.
        column_name (str): The name of the column to count the number of lines.

    Returns:
        DataFrame: The input DataFrame with an additional 'lines' column containing the number of lines.

    Raises:
        ValueError: If the specified column_name does not exist in the DataFrame.

    Example:
        df = pd.DataFrame({'lyrics': ['Line 1\nLine 2\nLine 3', 'Line 1\nLine 2']})
        number_of_lines(df, 'lyrics')
        # Resulting DataFrame:
        #                  lyrics  lines
        # 0  Line 1\nLine 2\nLine 3      3
        # 1         Line 1\nLine 2      2
    """

    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    df['lines'] = df[column_name].map(lambda t: len(re.findall(r'\n', t)))

    return df


def remove_new_lines(df, column_name):
    """
    Removes new lines from a specified column of a DataFrame.

    Args:
        df (DataFrame): The DataFrame to process.
        column_name (str): The name of the column to remove new lines from.

    Returns:
        DataFrame: The input DataFrame with new lines removed from the specified column.

    Raises:
        ValueError: If the specified column_name does not exist in the DataFrame.

    Example:
        df = pd.DataFrame({'lyrics': ['Line 1\nLine 2\nLine 3', 'Line 1\nLine 2']})
        remove_new_lines(df, 'lyrics')
        # Resulting DataFrame:
        #            lyrics
        # 0  Line 1Line 2Line 3
        # 1     Line 1Line 2
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    df[column_name] = df[column_name].map(lambda s: re.sub(r'\n|\n', ' ', s))

    return df




def tokenize_column(df, column_name, language='en'):
    """
    Tokenizes a specified column of a DataFrame based on the specified language.

    Args:
        df (DataFrame): The DataFrame to process.
        column_name (str): The name of the column to tokenize.
        language (str): The language of the column content ('en' for English, 'hy' for Armenian).

    Returns:
        DataFrame: The input DataFrame with a new column containing the tokenized values.

    Raises:
        ValueError: If the specified column_name does not exist in the DataFrame or the language is not supported.

    Example:
        df = pd.DataFrame({'lyrics': ['This is a sentence.', 'Այս հանդիպումը մի հատված է.']})
        tokenize_column(df, 'lyrics', 'en')
        # Resulting DataFrame:
        #                       lyrics              tokens_en
        # 0         This is a sentence.   [This, is, a, sentence, .]
        # 1  Այս հանդիպումը մի հատված է:  [Այս, հանդիպումը, մի, հատված, է, :]
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    if language not in ['en', 'hy']:
        raise ValueError("Unsupported language. Choose 'en' for English or 'hy' for Armenian.")
        
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    if language == 'en':
        df['tokens_en'] = df[column_name].map(tokenizer.tokenize)
    elif language == 'hy':
        df['tokens_hy'] = df[column_name].map(tokenizer.tokenize)

    return df




def lowercase_column(df, column_name):
    """
    Applies lowercase transformation to a specified column of a DataFrame.

    Args:
        df (DataFrame): The DataFrame to process.
        column_name (str): The name of the column to apply lowercase transformation.

    Returns:
        DataFrame: The input DataFrame with the specified column converted to lowercase.

    Raises:
        ValueError: If the specified column_name does not exist in the DataFrame.

    Example:
        df = pd.DataFrame({'lyrics': ['This is a sentence.', 'Այս հանդիպումը մի հատված է.']})
        lowercase_column(df, 'lyrics')
        # Resulting DataFrame:
        #                       lyrics
        # 0         this is a sentence.
        # 1  այս հանդիպումը մի հատված է.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    df[column_name] = df[column_name].apply(lambda tokens: [token.lower() for token in tokens])

    return df




from nltk.corpus import stopwords

nltk.download('stopwords')

def remove_stop_words(word_list,stop_words):
    filtered_words = [word for word in word_list if word not in stop_words]
    return filtered_words
def remove_stopwords(df, column_name, language='en'):
    """
    Removes stop words from a specified column of a DataFrame containing tokenized words.

    Args:
        df (DataFrame): The DataFrame to process.
        column_name (str): The name of the column containing tokenized words.
        language (str): The language of the tokenized words ('en' for English, 'hy' for Armenian).

    Returns:
        DataFrame: The input DataFrame with the specified column after removing stop words.

    Raises:
        ValueError: If the specified column_name does not exist in the DataFrame.
        ValueError: If the language is not 'en' or 'hy'.

    Example:
        df = pd.DataFrame({'tokens_en': [['this', 'is', 'a', 'sentence'], ['stop', 'words', 'removal']],
                           'tokens_hy': [['այս', 'հանդիպումը', 'մի', 'հատված', 'է'], ['կոռուպցիա', 'բարեկամություն', 'հեռանում']]}))
        remove_stopwords(df, 'tokens_en', 'en')
        # Resulting DataFrame:
        #               tokens_en             tokens_hy
        # 0     [sentence]       [այս, հանդիպումը, մի, հատված, է]
        # 1  [stop, words, removal]  [կոռուպցիա, բարեկամություն, հեռանում]
    """
    # Check if the column_name exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    # Check if the language is supported
    if language not in ['en', 'hy']:
        raise ValueError("Invalid language. Supported languages are 'en' (English) and 'hy' (Armenian).")

    # Get the appropriate stopwords based on the language
    if language == 'en':
        stop_words = set(stopwords.words('english'))
    elif language == 'hy':
        stop_words = [
    'այդ', 'այլ', 'այն', 'այս', 'դու', 'դուք', 'եմ', 'ես', 'են', 'ենք', 'եք', 'է',
    'էի', 'էին', 'էինք', 'էիր', 'էիք', 'էս', 'ըստ', 'թե', 'ի', 'ին', 'իսկ', 'իր', 'կամ',
    'համար', 'հետ', 'հետո', 'մենք', 'մեջ', 'մի', 'նա', 'նաև', 'նրա', 'որ', 'որը', 'որոնք',
    'որպես', 'ու', 'ում', 'պիտի', 'վրա', 'և','մի', 'ահա']
        
    df[column_name] = df[column_name].apply(lambda word_list: remove_stop_words(word_list, stop_words))
    return df


from nltk.stem import SnowballStemmer

def map_tokens_to_stems(df, column_name):
    """
    Maps tokens in the specified column to their stems using the SnowballStemmer.

    Args:
        df (DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column containing the tokens.

    Returns:
        DataFrame: The updated DataFrame with the 'stems_en' column.
    """
    # Create a stemmer object for English
    stemmer = SnowballStemmer('english')

    # Create a dictionary to map tokens to their stems
    token_to_stem = {}

    # Iterate through all songs
    for lst in df[column_name]:
        # Iterate through all tokens of the song
        for token in lst:
            # Check if the token is in the dictionary
            if token not in token_to_stem:
                # Add the token to the dictionary with its stem
                token_to_stem[token] = stemmer.stem(token)

    # Map the tokens to their stems and add the 'stems_en' column to the DataFrame
    df['stems_en'] = df[column_name].map(lambda lst: [token_to_stem[token] for token in lst])

    return df




def calculate_stem_word_count(df, column_name, new_column_name='n_stems_en'):
    """
    Calculates the number of stem words in the specified column and adds a new column with the count.

    Args:
        df (DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column containing the stem words.
        new_column_name (str): The name for the new column to be added.

    Returns:
        DataFrame: The updated DataFrame with the new column.
    """
    # Calculate the number of stem words for each row
    df[new_column_name] = df[column_name].map(len)

    return df

import random

def delete_songs_by_artist(df, artist_name, num_songs_to_delete):
    """
    Deletes a specified number of songs by the given artist from the DataFrame.

    Args:
        df (DataFrame): The DataFrame containing the songs.
        artist_name (str): The name of the artist whose songs will be deleted.
        num_songs_to_delete (int): The number of songs to delete.

    Returns:
        DataFrame: A DataFrame containing the deleted songs.

    Raises:
        ValueError: If the specified artist_name does not exist in the DataFrame.
        ValueError: If the number of songs to delete is greater than the total number of songs by the artist.

    Example:
        df = pd.DataFrame({'Song': ['Song 1', 'Song 2', 'Song 3', 'Song 4'],
                           'Artist': ['Artist A', 'Artist B', 'Artist A', 'Artist C']})

        deleted_songs = delete_songs_by_artist(df, 'Artist A', 2)
        # Resulting DataFrame (df):
        #    Song    Artist
        # 1  Song 2  Artist B
        # 3  Song 4  Artist C

        # Resulting DataFrame (deleted_songs):
        #    Song    Artist
        # 0  Song 1  Artist A
        # 2  Song 3  Artist A
    """
    # Check if the artist_name exists in the DataFrame
    if artist_name not in df['Artist'].unique():
        raise ValueError(f"Artist '{artist_name}' does not exist in the DataFrame.")

    # Filter the DataFrame to select songs by the specified artist
    artist_songs = df[df['Artist'] == artist_name]

    # Check if the number of songs to delete is valid
    if num_songs_to_delete > len(artist_songs):
        raise ValueError(f"The number of songs to delete is greater than the total number of songs by {artist_name}.")

    # Randomly select songs to delete
    songs_to_delete = random.sample(artist_songs.index.tolist(), num_songs_to_delete)

    # Create a new DataFrame to store the deleted songs
    deleted_songs = df.loc[songs_to_delete].copy()

    # Delete the selected songs from the original DataFrame
    df = df.drop(songs_to_delete)

    return df,deleted_songs


from sklearn.feature_extraction.text import TfidfVectorizer
def tf_idf_extraction(df):
    vectorizer = TfidfVectorizer()
    tf_idf_df=vectorizer.fit_transform(df)
    return tf_idf_df
    
