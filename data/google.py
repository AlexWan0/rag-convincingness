import requests
import json
from retry import retry
import pandas as pd
import os
import pickle
from tqdm import tqdm
import time

import logging
logging.basicConfig()


API_KEY = 'API_KEY'
CX_KEY = 'CX_KEY'

url_template = 'https://www.googleapis.com/customsearch/v1?key={key}&cx={cx}&q={query}'

MAX_QUERIES_PER_MINUTE = 100

@retry(tries=3, delay=120)
def _search(query: str, num_results=10) -> list[dict]:
    num_results = max(min(num_results, 100), 1)  # Ensure num_results is between 1 and 100
    results = []
    
    for start in range(1, num_results, 10):
        url_template = 'https://www.googleapis.com/customsearch/v1?key={key}&cx={cx}&q={query}&num=10&start={start}'
        url = url_template.format(key=API_KEY, cx=CX_KEY, query=query, start=start)

        res = requests.get(url)

        if res.status_code != 200:
            print(res.text)
            raise Exception(f'API request failed with status code: {res.status_code}')

        time.sleep(60 / MAX_QUERIES_PER_MINUTE)

        results.extend(json.loads(res.text).get('items', []))

    return results


def make_df_from_queries(
        queries: list[str],
        search_inputs: list[str],
        temp_file_path: str = None,
        num_results=20
    ) -> pd.DataFrame:

    rows = []

    if os.path.isfile(temp_file_path):
        print('temp file found, loading...')
        with open(temp_file_path, 'rb') as f_in:
            rows = pickle.load(f_in)
    
    seen_search_terms = set([row[1] for row in rows])

    assert len(queries) == len(search_inputs)

    for search_query, s_input in tqdm(zip(queries, search_inputs), total=len(queries)):
        if s_input in seen_search_terms:
            continue

        search_results = _search(s_input, num_results=num_results)

        for result in search_results:
            rows.append([
                search_query,
                s_input,
                result['link'] if 'link' in result else None,
                result['title'] if 'title' in result else None,
                result['snippet'] if 'snippet' in result else None,
                json.dumps(result)
            ])

        if temp_file_path is not None:
            with open(temp_file_path, 'wb') as f_out:
                pickle.dump(rows, f_out)

    df = pd.DataFrame(rows, columns=['search_query', 'search_engine_input', 'url', 'title', 'snippet', 'request_dump'])
    df = df.rename(columns={'index': 'query_index'})
    df['doc_id'] = df.index

    return df
