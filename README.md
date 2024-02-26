# What Evidence Do Language Models Find Convincing?
Retrieval augmented models use text from arbitrary online sources to answer user queries. These sources vary in formatting, style, authority, etc. Importantly, these sources also often conflict with each other. We release a dataset of webpages that give conflicting answers to questions to study how models resolve these conflicts.

# Data overview
Our data is available to download [here](https://drive.google.com/drive/folders/1fU52Jg6wVXCr6J03i0bRy7hA02iVbeS_?usp=sharing). It is a pickled Pandas dataframe. To load, run:

```python
import pandas as pd
pd.read_pickle('data.pkl')
```

The columns are as follows:
* **category**: Category of search query, used to seed diverse generations
* **search_query**: The user query to retrieve webpages for.
* **search_engine_input**: The affirmative and negative statements used to query the search API.
* **search_type**: Either `yes_statement` or `no_statement`. Indicates whether the website was retrieved by searching for an affirmative statement or a negative statement.
* **url**: The url of the website.
* **title**: The title of the website.
* **text_raw**: The raw output from `jusText`. Contains the text for the entire webpage.
* **text_window**: The 512-token window deemed most relevant to the search query.
* **stance**: The stance of the website, determined by an ensemble of `claude-instant-v1` and `GPT-4-1106-preview`.

# Code overview
1) `get_raw_data.py` queries GPT-4 to create candidate search queries from a list of categories and collects conflicting webpages for those queries.
2) `threshold_text.py` selects the best 512 token snippet of text for each webpage. `clean_threshold_text.py` removes sentences cut-off in the middle.

Next, we filter to deduplicate our data and remove ambiguous/mislabeled samples:
1) `filter_deup.py` removes duplicate search queries (using a manually specified list).
2) `get_synth_labels.py` collects synthetic labels for affirmative/negative stance using `claude-instant-v1` and `gpt-4-1106-preview`.
3) `filter_ensemble_preds.py` filters the dataset to only include samples that have synthetic labels consistent across the two models.

These features are used for the correlational experiments, probing for whether various in-the-wild text features are predictive of text-convincingness. `features/` contains the implementaions for various text features. Example usage can be found in `add_features.py`.

# Citation
Please consider citing our work if you found this code or our paper beneficial to your research.

```
@misc{wan2024evidence,
      title={What Evidence Do Language Models Find Convincing?}, 
      author={Alexander Wan and Eric Wallace and Dan Klein},
      year={2024},
      eprint={2402.11782},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
