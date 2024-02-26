import argparse
import spacy
nlp = spacy.load('en_core_web_lg')

import pandas as pd
from tqdm import tqdm
from pandarallel import pandarallel

from transformers import AutoModel, AutoTokenizer
import torch


def sent_tokenize(text: str) -> list[str]:
    doc = nlp(text)
    return [sent.text for sent in doc.sents]


def remove_incomplete_sentences(text: str) -> tuple[str, tuple[str]]:
    sents = sent_tokenize(text)

    removed_first, removed_second = None, None
    
    # make sure there's at least two sentences
    if len(sents) < 2:
        return text, (removed_first, removed_second)

    sents = [s for s in sents if s.strip() != '']

    # make sure first character of first sentence is uppercase and not punctuation
    first_char = sents[0].strip()[0]
    if (first_char.isalpha() and first_char.islower()) or (first_char in '.,;:!?'):
        removed_first = sents.pop(0)

    # make sure last sentence ends with punctuation
    last_char = sents[-1].strip()[-1]
    if last_char.isalnum():
        removed_second = sents.pop(-1)
    
    return ' '.join(sents), (removed_first, removed_second)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Remove incomplete sentences at the beginnign and end of the extracted text.'
    )

    parser.add_argument('file_path', type=str)
    parser.add_argument('out_file_path', type=str)

    args = parser.parse_args()

    pages_df = pd.read_pickle(args.file_path)
    
    if 'text_thresh_window_sent' not in pages_df.columns:
        pandarallel.initialize(progress_bar=True, nb_workers=8)
        pages_df['text_thresh_window_sent'] = pages_df['text_thresh_window'].parallel_apply(lambda x: remove_incomplete_sentences(x)[0])
    else:
        print('text_thresh_window_sent already in columns, skipping')
    
    model = AutoModel.from_pretrained("sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco").to('cuda')
    tokenizer = AutoTokenizer.from_pretrained("sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco")
    
    result_rows = []
    
    pbar = tqdm(total=len(pages_df))

    with torch.no_grad():
        for search_query, group_rows in pages_df.groupby('search_query'):
            sq_tokenized = tokenizer(search_query, add_special_tokens=True, return_tensors='pt', truncation=True).to('cuda')
            sq_encoded = model(**sq_tokenized)[0][:,0,:].squeeze(0).detach().cpu()
        
            for _, row in group_rows.iterrows():
                text = row['text_thresh_window_sent']

                tokenized = tokenizer(text, add_special_tokens=True, truncation=True, return_tensors='pt').to('cuda')
                encoded = model(**tokenized)[0][:,0,:].squeeze(0).detach().cpu()
                
                row['score_sent'] = torch.dot(sq_encoded, encoded).item()

                result_rows.append(row)

                pbar.update(1)
    
    pages_df = pd.DataFrame(result_rows)

    pages_df.to_pickle(args.out_file_path)
