import fr_core_news_sm
import string
import re

nlp = fr_core_news_sm.load()
PUNCTUATIONS = string.punctuation.replace("'", '').replace('[', '').replace(']', '')
PUNCTUATIONS = '[' + ' '.join(PUNCTUATIONS) + ' « »' + ']'
STOP_WORDS = nlp.Defaults.stop_words
ADDED_STOP_WORDS = [
    't',
    'y',
    "s'",
    "l'",
    's',
]  # stop words that I found to be useful to increase performance
for word in ADDED_STOP_WORDS:
    STOP_WORDS.add(word)


def custom_tokenizer(sentence, remove_duplicate=False):
    """Return the tokenized sentence (no punctuation/space/capital letter and optionally no duplicate)"""
    tokens = re.sub(PUNCTUATIONS, ' ', sentence.lower())
    tokens = tokens.replace('-', ' ').split()
    tokens_clean = []
    for token in tokens:
        if "'" in token:
            split = token.split("'")
            if split[0] != "":
                token = split[0] + "'"
                if not (token in STOP_WORDS):
                    tokens_clean.append(token)
            if split[1] != "":
                if not (split[1] in STOP_WORDS):
                    if split[1][-1] == 's':
                        token = split[1][:-1]
                    tokens_clean.append(split[1])
        else:
            if not (token in STOP_WORDS):
                if token[-1] == 's':
                    token = token[:-1]
                tokens_clean.append(token)
    if remove_duplicate:
        tokens_clean = list(set(tokens_clean))

    return tokens_clean
