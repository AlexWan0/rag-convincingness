import pandas as pd
import json

from utils import filter_pair_thresh


input_pages_df_path = './data/synth_label_data/pages_df.pkl'
output_pages_df_path = './data/synth_label_data/pages_df.pkl'

duplicates_path = './websites/duplicate_search_queries.json'


pages_df = pd.read_pickle(input_pages_df_path)

with open(duplicates_path, 'r') as f:
    duplicate_sq = set(json.load(f))

# filter to remove duplicate search queries
pages_df = pages_df[~pages_df['search_query'].isin(duplicate_sq)]

# keep only docs that have enough comparison docs
pages_df, _ = filter_pair_thresh(pages_df, min_pairings=5, answer_col='anthropic_author_stance_p')

pages_df.to_pickle(output_pages_df_path)
