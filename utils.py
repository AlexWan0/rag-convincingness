import pickle
import pandas as pd
import os
from tqdm import tqdm
from itertools import chain
from tqdm.auto import tqdm
import multiprocessing as mp
from typing import Union, Callable, Literal
from functools import partial
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from typing import Tuple
import re
from pathlib import Path


def enforce_df_cols(df: pd.DataFrame, column_names: list[str]):
    for col in column_names:
        assert col in df.columns

def parse_model_classifications(model_prediction: str, classes: list[str]) -> str:
    if model_prediction is None:
        return None

    classes = [c.lower() for c in classes]

    for c in classes:
        if c in model_prediction.lower():
            return c
    
    return None


def parse_model_classifications_seq(model_predictions: list[str], classes: list[str]) -> str:
    return list(map(partial(parse_model_classifications, classes=classes), model_predictions))


def filter_pair_thresh(
        df: pd.DataFrame,
        min_pairings: int,
        answer_col: str = 'anthropic_author_stance_p'
    ) -> tuple[pd.DataFrame, set[int]]:

    good_ids = set()
    for _, group in df.groupby('search_query'):
        yes_rows = group[group[answer_col] == 'yes']
        no_rows = group[group[answer_col] == 'no']

        num_yes = len(yes_rows)
        num_no = len(no_rows)

        if num_yes >= min_pairings or num_no >= min_pairings:
            good_ids.update(no_rows['answer_id'].tolist())
            good_ids.update(yes_rows['answer_id'].tolist())
    
    return df[df['answer_id'].isin(good_ids)], good_ids


def filter_by_id(
        pages_df: pd.DataFrame,
        pair_df: pd.DataFrame,
        ids: set[int],
        whitelist: bool=False
    ):
    
    if whitelist:
        pages_df = pages_df[pages_df['answer_id'].isin(ids)]
        pair_df = pair_df[
            pair_df['answer_id_1'].isin(ids) &
            pair_df['answer_id_2'].isin(ids)
        ]
    else:
        pages_df = pages_df[~pages_df['answer_id'].isin(ids)]
        pair_df = pair_df[
            (~pair_df['answer_id_1'].isin(ids)) &
            (~pair_df['answer_id_2'].isin(ids))
        ]
    
    return pages_df, pair_df


def longest_common_ngram(text: str, query: str) -> Tuple[str, ...]:
    # Instantiate a PorterStemmer
    ps = PorterStemmer()
    
    # Convert to lowercases and remove all non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    query = re.sub(r'[^a-zA-Z0-9\s]', ' ', query.lower())
    
    # Tokenize and stem the text and query
    text_tokens = [ps.stem(token) for token in word_tokenize(text)]
    query_tokens = [ps.stem(token) for token in word_tokenize(query)]
    
    # Initialize a 2D list (matrix) for dynamic programming
    dp = [[0] * (len(text_tokens) + 1) for _ in range(len(query_tokens) + 1)]
    
    # Initialize variables to store the length and position of the longest common n-gram
    max_length = 0
    end_pos = 0

    # Use dynamic programming to find the longest common n-gram
    for i in range(1, len(query_tokens) + 1):
        for j in range(1, len(text_tokens) + 1):
            if query_tokens[i - 1] == text_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_pos = i
            else:
                dp[i][j] = 0
                
    # Extract the longest common n-gram from the query tokens
    if max_length > 0:
        start_pos = end_pos - max_length
        longest_ngram = tuple(query_tokens[start_pos:end_pos])
        return longest_ngram
    else:
        return tuple()


def mkdir_recurse(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
