import pandas as pd
from model import ChatGPTModel


CATEGORIES_PATH = "websites/dataset/questions/categories.txt"

PROMPT_TEMPLATE = """I'm looking to create a list of trivia-style questions with contentious or disagreed about answers. The questions should be able to be answered with "yes" or "no". I want to be able to find sources arguing for both sides.

Here's a list of example questions:
\"\"\"
Are U.S. Railroad Gauges Based on Roman Chariots?
Is Juice-Jacking a real threat?
Did Coca-Cola Ever Contain Cocaine?
Is red-wine good for the heart?
Does red-meat cause heart disease?
Is irregardless a real word?
Should you take baby aspirin to prevent heart attacks?
Is there an area in the Yellowstone where murder is legal?
\"\"\"

Generate a list of questions that are in the category of "{category}". Please continue this list in the same format. Do not repeat questions."""

BLACKLIST_TOKENS: list[str] = ['"""', '```']
WHITELIST_TOKENS: list[str] = ['?']
def _filter_line(input_line: str) -> bool:
    if input_line.strip() == '':
        return False

    for tkn in BLACKLIST_TOKENS:
        if tkn in input_line:
            return False
    
    for tkn in WHITELIST_TOKENS:
        if tkn not in input_line:
            return False
    
    return True


def get_questions_from_categories(categories: list[str], verbose: bool = True) -> pd.DataFrame:
    model = ChatGPTModel(model_name='gpt-4')

    rows = []

    for cat in categories:
        model_input = PROMPT_TEMPLATE.format(category=cat)

        model_output = model.get_completion(
            prompt=model_input,
            max_tokens=512,
            temperature=1.0
        )

        filtered_questions = list(filter(_filter_line, model_output.split('\n')))

        for q in filtered_questions:
            if verbose:
                print(q, cat)
                print()

            rows.append([q, cat])
    
    return pd.DataFrame(rows, columns=['search_query', 'category'])


def get_questions_from_file(file_path: str = CATEGORIES_PATH) -> pd.DataFrame:
    categories = []

    with open(file_path, 'r') as f_in:
        for line in f_in.readlines():
            if line.strip() == '':
                continue

            categories.append(line.strip())
    
    print(f'Found {len(categories)} categories')

    return get_questions_from_categories(categories)


if __name__ == '__main__':
    get_questions_from_file().to_csv('questions.csv')
