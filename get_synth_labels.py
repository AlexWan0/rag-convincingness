from qa.single import add_author_query_prompt
import os
import pandas as pd


def run(input_pages_path, exp_path, model_name, pred_col):
    '''
    Add columns for synthetic labels using specified model.

    input_pages_path: path to pages_df.pkl without any labels
    exp_path: path to save pages_df.pkl with labels
    model_name: model name found in websites.model.base
    pred_col: name of column to save predictions to
    '''
    pages_df = pd.read_pickle(input_pages_path)

    pages_df = add_author_query_prompt(
        pages_df,
        exp_path=exp_path,
        text_col='text_thresh_window_sent',
        model_name=model_name,
        num_devices=10,
        verbose=True,
        catch_exceptions=False
    )

    pages_df = pages_df.rename(
        columns={'model_answer': pred_col}
    )

    pages_df.to_pickle(os.path.join(exp_path, 'pages_df.pkl'))


if __name__ == '__main__':
    # claude-instant-v1 synthetic labels
    run(
        input_pages_path='data/raw_data/pages_df.pkl',
        exp_path='./data/synth_label_data/claude',
        model_name='anthropic/claude-instant-v1',
        pred_col='anthropic_author_stance'
    )

    # gpt-4 synthetic labels
    run(
        input_pages_path='data/raw_data/pages_df.pkl',
        exp_path='./data/synth_label_data/gpt4',
        model_name='openai/gpt-4-1106-preview',
        pred_col='gpt-4_stance'
    )
