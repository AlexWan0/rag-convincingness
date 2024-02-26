import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
import pandas as pd

from features import Feature
from utils import enforce_df_cols


@torch.no_grad()
def forward_logprobs(text: str, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt').to('cuda')

    decoder_start_token_id = model.config.decoder_start_token_id
    decoder_input_ids = torch.ones((inputs.input_ids.shape[0], 1)) * decoder_start_token_id
    decoder_input_ids = decoder_input_ids.int().to('cuda')

    outputs = model(inputs.input_ids, decoder_input_ids=decoder_input_ids)

    logits = outputs.logits

    logprobs = torch.nn.functional.log_softmax(logits[0], dim=-1)

    return logprobs


def classify(prompt, text, labels, tokenizer, model):
    label_tkns = tokenizer(labels, add_special_tokens=False).input_ids
    label_ids = [l[0] for l in label_tkns]

    logprobs = forward_logprobs(
        prompt.format(text=text),
        tokenizer,
        model
    )
    
    logprobs = [logprobs[0, label_id].item() for label_id in label_ids]

    return {
        label: logprob
        for label, logprob in zip(labels, logprobs)
    }


def sentiment_classify(text, tokenizer, model):
    return classify(
        """Is the sentiment of the following text Positive or Negative?\nText:{text}""",
        text,
        ["Positive", "Negative"],
        tokenizer,
        model
    )

class SentimentFLAN(Feature):
    col_name = '_sentiment_flan'

    def __call__(self, pages_df: pd.DataFrame) -> pd.DataFrame:
        enforce_df_cols(pages_df, [self.text_col])

        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
        model = model.to('cuda')

        sentiment_classify_fn = lambda x: sentiment_classify(x, tokenizer, model)

        pages_df[self.col_name] = pages_df[self.text_col].progress_apply(sentiment_classify_fn)

        return pages_df
