import pandas as pd
import os
from collections.abc import Callable
from tqdm import tqdm

from web_engines import WebEngine


def basic_filtering(page_text: str) -> bool:
    if page_text is None:
        return False

    num_tokens = len(page_text.split(' '))

    if num_tokens < 200:
        return False
    
    return True


def add_text_col(
        df: pd.DataFrame,
        web_engine: WebEngine,
        website_folder: str,
        source_process_func: Callable[[str], str],
        skip_web: bool = False
    ):
    if not os.path.exists(website_folder):
        os.mkdir(website_folder)
    
    # first pass, download websites to disk
    # add source_path col
    paths = []
    total = len(df)
    for _, row in tqdm(df.iterrows(), total=total):
        doc_id = row['doc_id']
        url = row['url']

        print(doc_id, url)

        output_path = os.path.join(website_folder, f'{doc_id}.html')        

        # try and download source
        source = None
        if os.path.exists(output_path):
            print('exists')
            with open(output_path) as f_in:
                source = f_in.read()
        elif not url.endswith('.pdf') and not skip_web:
            try:
                source = web_engine.request_get(url)
            except Exception as e:
                print(e)

        # save source if successful
        if source is not None:
            with open(output_path, 'w') as f_out:
                f_out.write(source)

            paths.append(output_path)
        else:
            paths.append(None)
    
    df['source_path'] = paths

    # run source preprocessing
    def _load_and_process(source_path):
        if source_path is None:
            return None

        with open(source_path) as f_in:
            source = f_in.read()

        source = source_process_func(source)

        return source
    
    df['text'] = df['source_path'].apply(_load_and_process)

    df = df[df['text'].apply(basic_filtering)]

    return df
