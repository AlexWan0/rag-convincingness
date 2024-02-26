import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd

from utils import enforce_df_cols
from features import Feature


def compute_perplexity(text, model, tokenizer):
    # Tokenize the input text
    input_ids = tokenizer.encode(text, return_tensors='pt').to('cuda')
    
    # Compute the log-likelihood of the input text
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        log_likelihood = outputs[0].item()

    # Compute perplexity as exp(-log_likelihood/num_tokens)
    perplexity = torch.exp(torch.tensor(log_likelihood))
    
    return perplexity.item()


class PerplexityGPT2(Feature):
    col_name = '_perplexity_gpt2'

    def __call__(self, pages_df: pd.DataFrame) -> pd.DataFrame:
        enforce_df_cols(pages_df, [self.text_col])

        # Load pre-trained model and tokenizer
        model = GPT2LMHeadModel.from_pretrained('gpt2-medium').to('cuda')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')

        perplexity_fn = lambda x: compute_perplexity(x, model, tokenizer)

        pages_df[self.col_name] = pages_df[self.text_col].progress_apply(perplexity_fn)

        return pages_df
