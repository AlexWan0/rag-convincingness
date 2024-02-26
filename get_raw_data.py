import os
import pandas as pd

from data.questions import get_questions_from_file
from data.query_generator import get_yes_no_pairs, quote_shared_tokens
from data.google import make_df_from_queries
import data.web as web
from data.web_engines import RequestsWebEngine
from data.text_processing import justext_html_to_text


def build_dataset(exp_path: str, questions_path: str) -> pd.DataFrame:  
    '''
    Converts a csv of questions into a dataset of webpages, retrieved using google searches.
    ''' 

    query_temp_path = os.path.join(exp_path, '_temp_query_df.pkl')

    if os.path.exists(query_temp_path):
        print('loading from file: ', query_temp_path)
        pages_df = pd.read_pickle(query_temp_path)
    else:
        # two cols: search_query, category
        pages_df = pd.read_csv(os.path.join(questions_path), sep=',')

        yes_statements, no_statements = get_yes_no_pairs(pages_df['search_query'].tolist())

        pages_df['yes_statement'] = yes_statements
        pages_df['no_statement'] = no_statements

        pages_df = pages_df.melt(
            id_vars=['category', 'search_query'],
            value_vars=['yes_statement', 'no_statement'],
            var_name='search_type',
            value_name='search_engine_input'
        )

        pages_df.to_pickle(query_temp_path)

    def _quote_shared_tokens(row):
        row['search_engine_input'] = quote_shared_tokens(row['search_engine_input'], row['search_query'])[0]
        return row

    pages_df = pages_df.apply(_quote_shared_tokens, axis=1)

    search_df = make_df_from_queries(
        pages_df['search_query'].tolist(),
        pages_df['search_engine_input'].tolist(),
        temp_file_path=os.path.join(exp_path, '_temp_search.pkl'),
        num_results=20
    )

    pages_df = pages_df.merge(search_df, on=['search_query', 'search_engine_input'], validate='1:m')

    with RequestsWebEngine() as r_engine:
        pages_df = web.add_text_col(
            pages_df,
            web_engine=r_engine,
            website_folder=os.path.join(exp_path, 'pages'),
            source_process_func=justext_html_to_text,
        )

    return pages_df


if __name__ == '__main__':
    get_questions_from_file().to_csv('data/raw_data/questions.csv')

    build_dataset(
        'data/raw_data',
        'data/raw_data/questions.csv'
    ).to_pickle('data/raw_data/pages_df.pkl')
