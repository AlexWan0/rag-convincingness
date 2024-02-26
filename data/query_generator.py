from tqdm import tqdm
from retry import retry

from model import Model, ChatGPTModel
import spacy
import re
from nltk.stem import PorterStemmer
from nltk.corpus import words
from tqdm import tqdm


prompt = '''Question: {question}

This is a yes-or-no question. Rewrite this question as a statement, first in the affirmative and then in the negative.
Write multiple versions of each statement, varying the wording and phrasing.

Use the following format for your answer:
Yes 1: Statement where answer is Yes
Yes 2: Another statement where answer is Yes
No 1: Statement where answer is No
No 2: Another statement where answer is No

Example input:
Question: Is red wine good for the heart?

Example answer:
Yes: Red Wine Is Good for the Heart
No: Red Wine Is Not Good for the Heart'''

@retry(tries=3, delay=30)
def _get_pair(model: Model, question: str) -> dict[str, str]:
    model_response = model.get_completion(
        prompt.format(question=question)
    )

    yes = []
    no = []
    for resp in model_response.split('\n'):
        if resp.lower().startswith('yes'):
            # yes.append(resp.replace('Yes: ', ''))
            yes.append(':'.join(resp.split(':')[1:]).strip())
        elif resp.lower().startswith('no'):
            # no.append(resp.replace('No: ', ''))
            no.append(':'.join(resp.split(':')[1:]).strip())
    
    yes = [x.strip() for x in yes if x.strip() != '']
    no = [x.strip() for x in no if x.strip() != '']

    yes = '. '.join(yes)
    no = '. '.join(no)

    return {'yes': yes, 'no': no}


def get_yes_no_pairs(questions: list[str]) -> tuple[list[str], list[str]]:
    model = ChatGPTModel(model_name='gpt-3.5-turbo')

    yes_statements = []
    no_statements = []

    for i, q in enumerate(tqdm(questions)):
        try:
            response = _get_pair(model, q)
            yes_statements.append(response['yes'])
            no_statements.append(response['no'])
        except Exception as e:
            print(i, e)
            yes_statements.append(None)
            no_statements.append(None)

    return yes_statements, no_statements


nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()


def get_stemp_to_words() -> dict[str, list[str]]:
    stem_to_words = {}

    for word in tqdm(words.words()):
        stem = stemmer.stem(word)
        if stem not in stem_to_words:
            stem_to_words[stem] = []
        stem_to_words[stem].append(word.lower())
    
    return stem_to_words


def quote_shared_tokens(str1: str, str2: str) -> tuple[str, str]:
    # Tokenize the strings and get stems for non-stop words
    doc1 = nlp(str1)
    doc2 = nlp(str2)

    tokens1 = [(token.text, stemmer.stem(token.text)) for token in doc1 if not token.is_stop]
    tokens2 = [(token.text, stemmer.stem(token.text)) for token in doc2 if not token.is_stop]

    # Get set of stems
    stems1 = set(stem for _, stem in tokens1)
    stems2 = set(stem for _, stem in tokens2)

    # Find shared stems
    shared_stems = stems1.intersection(stems2)

    # Find original tokens associated with shared stems
    shared_tokens = [(token, stem) for token, stem in tokens1 + tokens2 if stem in shared_stems]

    # Replace shared tokens with quoted versions in original sentences
    for token, stem in shared_tokens:
        pattern = r'\b' + re.escape(token) + r'\b'  # \b indicates word boundary, re.escape() to escape special characters
        
        # exploded_set = set([token.lower()] + stem_to_words[stem][:2]) if stem in stem_to_words else set([token.lower()])
        # replacement = '|'.join([f'"{t}"' for t in exploded_set])
        replacement = f'"{token}"'
        
        str1 = re.sub(pattern, replacement, str1)
        str2 = re.sub(pattern, replacement, str2)

        str1 = str1.lower()
        str2 = str2.lower()

    str1 = re.sub(r'"{1,}', '"', str1)
    str2 = re.sub(r'"{1,}', '"', str2)

    return str1, str2
