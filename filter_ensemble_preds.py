import pandas as pd
import json

from utils import parse_model_classifications_seq, filter_pair_thresh


output_path = './websites/ensemble_matched_ids.json'
output_path_pages_df = './data/synth_label_data/pages_df.pkl'

synth_paths = {
    'claude': './data/synth_label_data/claude/pages_df.pkl',
    'gpt4': './data/synth_label_data/gpt4/pages_df.pkl'
}

# the labels given by claude get added to the pages_df as the 'anthropic_author_stance' column
# the labels given by gpt-4 get added to the pages_df as the 'gpt-4_stance' column
model_answer_cols = {
    'claude': 'anthropic_author_stance',
    'gpt4': 'gpt-4_stance'
}

# load synthetic labels and rename cols to match
synth_dfs = {
    k: pd.read_pickle(v)[['answer_id', model_answer_cols[k]]].rename(
        columns={model_answer_cols[k]: 'model_answer'}
    )
    for k, v in synth_paths.items()
}

# parse gpt predictions
def split_lastline(text):
    if text is None:
        return None
    return text.split('\n')[-1]

pages_df_gpt4 = synth_dfs['gpt4']

pages_df_gpt4['last_line'] = pages_df_gpt4['model_answer'].apply(split_lastline)

pages_df_gpt4['gpt_4_p'] = parse_model_classifications_seq(pages_df_gpt4['last_line'], ['not', 'yes', 'no'])

# parse claude predictions
pages_df_claude = synth_dfs['claude']

pages_df_claude['anthropic_p'] = parse_model_classifications_seq(pages_df_claude['model_answer'], ['yes', 'no'])

# find matching answers
merge_df = pages_df_claude[['answer_id', 'anthropic_p']].merge(
    pages_df_gpt4[['answer_id', 'gpt_4_p']],
    on='answer_id'
)

ensemble_matched_ids = merge_df[merge_df['anthropic_p'] == merge_df['gpt_4_p']]['answer_id'].tolist()

with open(output_path, 'w') as f:
    json.dump(ensemble_matched_ids, f)

# save filtered pages_df
df = pd.read_pickle(synth_paths['claude'])

# parse claude predictions
df['anthropic_author_stance_p'] = parse_model_classifications_seq(pages_df_claude['model_answer'], ['yes', 'no'])

print(len(df))

# filter by ensemble ids
df = df[df['answer_id'].isin(ensemble_matched_ids)]

print(len(df))

# keep only docs that have enough comparison docs
df, _ = filter_pair_thresh(df, min_pairings=5, answer_col='anthropic_author_stance_p')

print(len(df))

df.to_pickle(output_path_pages_df)
