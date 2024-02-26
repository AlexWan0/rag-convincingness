import pandas as pd
from colors import color

from features.features import PerplexityGPT2, NumTokens, NumUniqueTokens, FleschKincaidReadibility, MaxNgramOverlap, SentimentFLAN, STSimilarity


orig_path = 'data/synth_label_data/pages_df.pkl'
output_path = 'data/features/pages_df.pkl'
text_col = 'text_thresh_window_sent'

if __name__ == '__main__':
    pages_df = pd.read_pickle(orig_path)

    pages_df = pages_df[[text_col, 'search_query']]

    features = [
        STSimilarity(text_col = text_col),
        NumTokens(text_col = text_col),
        NumUniqueTokens(text_col = text_col),
        FleschKincaidReadibility(text_col = text_col),
        MaxNgramOverlap(text_col = text_col),
        SentimentFLAN(text_col = text_col),
        PerplexityGPT2(text_col = text_col)
    ]

    for feature in features:
        print(color('Running feature: %s' % feature.__class__.__name__, bg='blue', fg='white'))
        feature(pages_df)

    pages_df.to_pickle(output_path)
