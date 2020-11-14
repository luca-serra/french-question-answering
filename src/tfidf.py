"""
Implementing a TFIDF approach to find the best context
"""

import json
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import utils
import fr_core_news_sm

nlp = fr_core_news_sm.load()

stop_words = nlp.Defaults.stop_words

with open('../datasets/data/contexts.json', encoding="utf8") as json_file:
    contexts = json.load(json_file)

with open('../datasets/data/questions.json', encoding="utf8") as json_file:
    questions = json.load(json_file)

# for question in questions['data']:
start_time = time.time()

question = questions['data'][0]['question']
context_id = questions['data'][0]['context_id']

# tokenizer = TfidfVectorizer(lowercase=True).build_tokenizer()
tokenizer = utils.spacy_tokenizer
vocabulary = tokenizer(question)
vectorizer = TfidfVectorizer(vocabulary=vocabulary, stop_words=stop_words)
# print([tuple['context'] for tuple in contexts['data'][:5]])
x = vectorizer.fit_transform([question] + [tuple['context'] for tuple in contexts['data'][:]])
x = x.toarray()
# features = np.zeros((3, 2))
# print(features)


def build_features(row):
    """Given a row of the TF-IDF matrix, return (mean, ratio_complete) of the row"""
    c = 0
    for value in row:
        if value != 0:
            c += 1

    return [np.mean(row), c / row.shape[0]]


features = []
for row_idx in range(x.shape[0]):
    features.append(build_features(x[row_idx, :]))

features = np.array(features)

print(vectorizer.get_feature_names())
# print(x)
print('idf')
print(features)
print(np.argmax(features[1:, 0]))
print(np.argmax(features[1:, 1]))
print(vocabulary)
print(f'{time.time() - start_time:.2f} sec')


# ['test de mon algorithme', 'ceci est un test à tester', "Ma phrase de test doit être testée (et non pas tester) comme ceci, j'attends le résultat. Chanteur chanter chant ! Petit test; OK. L'apostrophe"]
