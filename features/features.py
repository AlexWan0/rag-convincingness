import pandas as pd
from utils import enforce_df_cols, longest_common_ngram
import re
from typing import Callable

from transformers import AutoTokenizer, AutoModel
from readability import Readability
from readability.exceptions import ReadabilityException
import tldextract
from enum import Enum
from pandarallel import pandarallel
import os
from nltk.stem import WordNetLemmatizer
import pickle

from tqdm.auto import tqdm
tqdm.pandas()

import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')
nlp.max_length = 100_000_000

from sentiment import SentimentFLAN
from gpt2_perplexity import PerplexityGPT2


class Feature():
    col_name: str

    def __init__(self, text_col: str = 'text', temp_path: str = './', num_workers: int = 4, verbose: bool = False):
        self.text_col = text_col

        self.num_workers = num_workers

        self.verbose = verbose

        self.temp_path = temp_path

        if not self.col_name.startswith('_'):
            raise ValueError('col_name must start with an underscore')
    
    def __call__(self, pages_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError('Feature must be implemented')


class NumTokens(Feature):
    col_name = '_num_tokens'

    def __call__(self, pages_df: pd.DataFrame) -> pd.DataFrame:
        enforce_df_cols(pages_df, [self.text_col])

        # basic tokens
        pages_df[self.col_name] = pages_df[self.text_col].progress_apply(
            lambda x: len([tkn for tkn in x.split(' ') if tkn.split() != ''])
        )
        
        # spacy tokens
        # pages_df['num_tokens'] = pages_df['text'].progress_apply(lambda x: len([tkn for tkn in nlp(x) if tkn.text.split() != '']))
        
        return pages_df


class NumUniqueTokens(Feature):
    col_name = '_num_unique_tokens'

    def __call__(self, pages_df: pd.DataFrame) -> pd.DataFrame:
        enforce_df_cols(pages_df, [self.text_col])

        wnl = WordNetLemmatizer()

        # basic tokens
        pages_df[self.col_name] = pages_df[self.text_col].progress_apply(
            lambda x: len(set([wnl.lemmatize(tkn) for tkn in x.split(' ') if tkn.split() != '']))
        )
        
        # spacy tokens
        # pages_df['num_tokens'] = pages_df['text'].progress_apply(lambda x: len([tkn for tkn in nlp(x) if tkn.text.split() != '']))
        
        return pages_df


class HrefCount(Feature):
    col_name = '_href_count'

    def __call__(self, pages_df: pd.DataFrame) -> pd.DataFrame:
        enforce_df_cols(pages_df, [self.text_col])

        pandarallel.initialize(progress_bar=True, nb_workers=self.num_workers)
        pages_df[self.col_name] = pages_df[self.text_col].parallel_apply(lambda x: len(re.findall(r'href\s*=', x)))

        return pages_df


# https://gist.github.com/gruber/8891611
url_regex_pattern = r'(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))'

class UrlCount(Feature):
    col_name = '_url_count'
        
    def __call__(self, pages_df: pd.DataFrame) -> pd.DataFrame:
        enforce_df_cols(pages_df, [self.text_col])

        pandarallel.initialize(progress_bar=True, nb_workers=self.num_workers)
        # use url_regex_pattern to count the number of urls in the text
        pages_df[self.col_name] = pages_df[self.text_col].parallel_apply(lambda x: len(re.findall(url_regex_pattern, x)))

        return pages_df


# https://stackoverflow.com/a/17681902
email_regex_pattern = r"([-!#-'*+/-9=?A-Z^-~]+(\.[-!#-'*+/-9=?A-Z^-~]+)*|\"([]!#-[^-~ \t]|(\\[\t -~]))+\")@([-!#-'*+/-9=?A-Z^-~]+(\.[-!#-'*+/-9=?A-Z^-~]+)*|\[[\t -Z^-~]*])"

class EmailCount(Feature):
    col_name = '_email_count'
        
    def __call__(self, pages_df: pd.DataFrame) -> pd.DataFrame:
        enforce_df_cols(pages_df, [self.text_col])

        pandarallel.initialize(progress_bar=True, nb_workers=self.num_workers)
        # use email_regex_pattern to count the number of emails in the text
        pages_df[self.col_name] = pages_df[self.text_col].parallel_apply(lambda x: len(re.findall(email_regex_pattern, x)))

        return pages_df


class MaxNgramOverlap(Feature):
    col_name = '_max_ngram_overlap'

    def __call__(self, pages_df: pd.DataFrame) -> pd.DataFrame:
        enforce_df_cols(pages_df, [self.text_col, 'search_query'])

        # pandarallel.initialize(progress_bar=True, nb_workers=self.num_workers)
        # use longest_common_ngram to find the longest ngram between the text and the search query
        pages_df[self.col_name] = pages_df.progress_apply(lambda x: len(longest_common_ngram(x[self.text_col], x['search_query'])), axis=1)

        return pages_df
    

class TLD(Feature):
    col_name = '_tld'

    def __call__(self, pages_df: pd.DataFrame) -> pd.DataFrame:
        enforce_df_cols(pages_df, ['url'])

        pandarallel.initialize(progress_bar=True, nb_workers=self.num_workers)
        pages_df[self.col_name] = pages_df['url'].parallel_apply(lambda x: tldextract.extract(x).suffix)

        return pages_df


class Domain(Feature):
    col_name = '_domain'

    def __call__(self, pages_df: pd.DataFrame) -> pd.DataFrame:
        enforce_df_cols(pages_df, ['url'])

        pandarallel.initialize(progress_bar=True, nb_workers=self.num_workers)
        pages_df[self.col_name] = pages_df['url'].parallel_apply(lambda x: tldextract.extract(x).domain)

        return pages_df


class FleschKincaidReadibility(Feature):
    col_name = '_flesch_kincaid_readibility'

    def __call__(self, pages_df: pd.DataFrame) -> pd.DataFrame:
        enforce_df_cols(pages_df, [self.text_col])

        def _wrapped_readability(text: str) -> float:
            try:
                return Readability(text).flesch_kincaid().score
            except ReadabilityException as e:
                print('ReadabilityException:', e)
                return 0.0

        pages_df[self.col_name] = pages_df[self.text_col].progress_apply(_wrapped_readability)

        return pages_df

class STSimilarity(Feature):
    col_name = '_st_similarity'

    def __init__(self, *args, device='cuda', **kwargs):
        model_name = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name) 
        self.bert_model = AutoModel.from_pretrained(model_name)

        self.bert_model = self.bert_model.to(device)
        self.device = device

        super().__init__(*args, **kwargs)

    def __call__(self, pages_df: pd.DataFrame) -> pd.DataFrame:
        enforce_df_cols(pages_df, [self.text_col, 'search_query'])

        def _st_sim(text: str, query: str) -> float:
            text_tkn = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
            query_tkn = self.tokenizer(query, return_tensors="pt").to(self.device)

            text_encoded = self.bert_model(**text_tkn)[0][:,0,:].squeeze(0)
            query_encoded = self.bert_model(**query_tkn)[0][:,0,:].squeeze(0)

            return query_encoded.dot(text_encoded).cpu().item()

        pages_df[self.col_name] = pages_df.progress_apply(lambda x: _st_sim(x[self.text_col], x['search_query']), axis=1)

        return pages_df


def collect_feature_subclasses() -> list[Feature]:
    return Feature.__subclasses__()

def add_all_features(pages_df: pd.DataFrame, skip: bool = True, text_col: str = 'text') -> pd.DataFrame:
    print(f'found {len(collect_feature_subclasses())} features')
    for f in collect_feature_subclasses():
        print(f'adding feature {f.col_name}')

        if skip and f.col_name in pages_df.columns:
            print(f'skipping feature {f.col_name}')
            continue
        
        try:
            pages_df = f(text_col = text_col)(pages_df)
        except Exception as e:
            print('Feature extraction error:', e)
    
    return pages_df


def get_paired_data(pairs_df: pd.DataFrame, col_name: str) -> dict[str, list]:
    values = []
    pick_probs = []

    for _, row in pairs_df.iterrows():
        values.append(row[f'{col_name}_1'] - row[f'{col_name}_2'])
        pick_probs.append(row['eval_1_prob'] - row['eval_2_prob'])

    return {'x': values, 'y': pick_probs}


def get_singular_data(pairs_df: pd.DataFrame, col_name: str) -> dict[str, list]:
    values = []
    pick_probs = []

    for _, row in pairs_df.iterrows():

        values.append(row[f'{col_name}_1'])
        values.append(row[f'{col_name}_2'])

        pick_probs.append(row['eval_1_prob'])
        pick_probs.append(row['eval_2_prob'])
    
    return {'x': values, 'y': pick_probs}
