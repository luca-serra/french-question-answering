"""
Implementing a TFIDF approach to find the best context
"""

import json
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import utils
import fr_core_news_sm
import random

nlp = fr_core_news_sm.load()

# stop_words = nlp.Defaults.stop_words

with open('../datasets/data/contexts.json', encoding="utf8") as json_file:
    contexts = json.load(json_file)

with open('../datasets/data/questions.json', encoding="utf8") as json_file:
    questions = json.load(json_file)


def build_features(row):
    """Given a row of the TF-IDF matrix, return (mean, ratio_complete) of the row"""
    c = 0
    for value in row:
        if value != 0:
            c += 1

    return [np.mean(row), c / row.shape[0]]


context_ids = [tuple['context_id'] for tuple in contexts['data']]

n = 100
c = 0
c_2 = 0
c_top_avg = 0
c_top_ratio = 0
inter = 0
random.shuffle(questions['data'])
for q in questions['data'][:n]:
    pred_true = False

    start_time = time.time()

    question = q['question']
    context_id = q['context_id']
    corpus = [question] + [tuple['context'] for tuple in contexts['data'][:]]
    # tokenizer = TfidfVectorizer(lowercase=True).build_tokenizer()
    # tokenizer = utils.spacy_tokenizer
    tokenizer = utils.custom_tokenizer
    vocabulary = tokenizer(question, remove_duplicate=True)
    vectorizer = TfidfVectorizer(vocabulary=vocabulary, tokenizer=tokenizer, norm=None, binary=True)
    # print([tuple['context'] for tuple in contexts['data'][:5]])
    x = vectorizer.fit_transform(corpus)
    x = x.toarray()
    # features = np.zeros((3, 2))
    # print(features)

    features = []
    for row_idx in range(x.shape[0]):
        features.append(build_features(x[row_idx, :]))

    features = np.array(features)
    print(question)
    print(vocabulary)
    # print(features[:5, :])
    indices_avg = np.argpartition(features[1:, 0], -5)[-5:]
    indices_ratio = np.argpartition(features[1:, 1], -5)[-5:]
    predicted_contexts_avg = [contexts['data'][idx]['context_id'] for idx in indices_avg]
    predicted_contexts_ratio = [contexts['data'][idx]['context_id'] for idx in indices_ratio]
    predicted_context = contexts['data'][np.argmax(features[1:, 0])]['context_id']
    predicted_context_2 = contexts['data'][np.argmax(features[1:, 1])]['context_id']
    # print(predicted_context)
    # print(predicted_context_2)
    # print(predicted_contexts_avg)
    # print(predicted_contexts_ratio)
    # print(context_id)
    if context_id == predicted_context:
        c += 1
        pred_true = True
    if context_id == predicted_context_2:
        c_2 += 1
        pred_true = True
    if pred_true:
        inter += 1
    if context_id in predicted_contexts_avg:
        print('OK')
        c_top_avg += 1
    else:
        print(predicted_contexts_avg)
        for idx in indices_avg:
            print(x[1 + idx, :])
        print(context_id)
        print(x[1 + context_ids.index(context_id), :])
    if context_id in predicted_contexts_ratio:
        c_top_ratio += 1
    print(f'{time.time() - start_time:.2f} sec')

print(
    f'n: {n}, ratio_avg: {c/n}, ratio_custom:{c_2/n}, ratio_inter: {inter/n}, accuracy top5 avg: {c_top_avg/n}, accuracy top5 ratio: {c_top_ratio/n}'
)

# ['test de mon algorithme', 'ceci est un test à tester', "Ma phrase de test doit être testée (et non pas tester) comme ceci, j'attends le résultat. Chanteur chanter chant ! Petit test; OK. L'apostrophe"]
