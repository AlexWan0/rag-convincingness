from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase, PreTrainedModel, AutoTokenizer, AutoModel
from itertools import islice
import torch
import argparse
import pandas as pd
from tqdm import tqdm
import pickle
import multiprocessing as mp


@dataclass
class Models:
    # used for limiting lengths
    target_tokenizer: PreTrainedTokenizerBase
    
    # used for encoding
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase


def init_models(device: int) -> Models:
    result = Models(
        target_tokenizer=AutoTokenizer.from_pretrained("lmsys/vicuna-13b-v1.5-16k"),
        model=AutoModel.from_pretrained("sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco").to(device),
        tokenizer=AutoTokenizer.from_pretrained("sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco")
    )

    return result


@torch.no_grad()
def find_best_window(
        text: str,
        search_query: str,
        models: Models,
        device: int,

        chunk_size: int,
        step_size: int,

        batch_size: int = 32,
        special_token_padding_size: int = 4
    ) -> tuple[str, float]:

    tokenized = models.target_tokenizer(text, add_special_tokens=False, return_tensors='pt')
    input_ids = tokenized['input_ids'][0, :]

    # make windows
    if input_ids.shape[0] < (chunk_size - special_token_padding_size):
        windows = input_ids.unsqueeze(0)
    else:
        windows = input_ids.unfold(0, chunk_size - special_token_padding_size, step_size)

    # encode query
    sq_tokenized = models.tokenizer(search_query, return_tensors='pt', truncation=True).to(device)
    sq_encoded = models.model(**sq_tokenized)[0][:,0,:].squeeze(0).detach().cpu()

    # convert tokens back to string
    windows_text = []
    for i in range(windows.shape[0]):
        untokenized = models.target_tokenizer.decode(windows[i, :])
        windows_text.append(untokenized)
    
    # encode windows
    windows_encodings_lst = []
    windows_iter = iter(windows_text)
    while text_batch := list(islice(windows_iter, batch_size)):
        bert_tokenized = models.tokenizer(
            text_batch,
            add_special_tokens=True,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=None # truncate to maximum length of model
        ).to(device)

        batch_encodings = models.model(**bert_tokenized)[0][:,0,:]

        windows_encodings_lst.append(batch_encodings.detach().cpu())

    windows_encodings = torch.cat(windows_encodings_lst, dim=0)

    assert len(windows_encodings) == len(windows_text)

    # get best window
    scores = torch.matmul(sq_encoded, windows_encodings.T)
    best_idx = torch.argmax(scores)
    best_score = scores[best_idx]

    return windows_text[best_idx], best_score.item()


def run(args, device: int, pages_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    results_score = []

    for i, (_, row) in enumerate(tqdm(pages_df.iterrows(), total=len(pages_df), desc=f'Device {device}', position=device)):
        text_orig = row['text']
        search_query = row['search_query']

        if not isinstance(text_orig, str) or not isinstance(search_query, str):
            results.append(None)
            results_score.append(None)
            continue

        models = init_models(device)

        text_short, score = find_best_window(
            text_orig,
            search_query,
            models,
            device,
            args.max_length,
            args.step_size,
            batch_size=32,
            special_token_padding_size=4
        )

        results.append(text_short)
        results_score.append(score)

        if i % 100 == 0:
            with open(f'_thresh_window_temp_{device}.pkl', 'wb') as f_out:
                pickle.dump(results, f_out)

    pages_df['text_thresh_window'] = results
    pages_df['text_thresh_window_score'] = results_score

    return pages_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Find the most relevant span of text in a page using sentence transformer scores.'
    )

    parser.add_argument('file_path', type=str)
    parser.add_argument('out_file_path', type=str)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--step_size', type=int, default=256)
    parser.add_argument('--num_devices', type=int, default=1)

    args = parser.parse_args()

    pages_df = pd.read_pickle(args.file_path)
    
    mp_args = []
    for device in range(args.num_devices):
        start_idx, end_idx = device * len(pages_df) // args.num_devices, (device + 1) * len(pages_df) // args.num_devices

        if device == args.num_devices - 1:
            end_idx = len(pages_df)
        
        mp_args.append((args, device, pages_df.iloc[start_idx:end_idx].copy()))
    
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.num_devices) as pool:
        results = pool.starmap(run, mp_args)
    
    pages_df = pd.concat(results)

    pages_df.to_pickle(args.out_file_path)
