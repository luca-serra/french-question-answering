"""
Implementing a TFIDF approach to find the best context for a given question
"""
import json
import random
import time

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import custom_tokenizer, utils
from illuin.src.utils import file_creation


class TfidfClassifier(BaseEstimator):
    def __init__(self, filename):
        """filename is the relative path to the JSON file on the FQuAD dataset format"""
        self.filename = filename

    def fit(self, X, y=None):
        return self  # for TFIDF model, there is no fit step

    def predict(self, file=True, n=-1, question=None):
        if file:
            questions, contexts = file_creation.build_questions_and_contexts(self.filename)
        else:
            _, contexts = file_creation.build_questions_and_contexts(self.filename)
            questions = [{'question': question, 'question_id': 0, 'context_id': None}]


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
    tokenizer = custom_tokenizer.custom_tokenizer
    vocabulary = tokenizer(question, remove_duplicate=True)
    vectorizer = TfidfVectorizer(
        vocabulary=vocabulary, tokenizer=tokenizer, norm=None, binary=True
    )
    # print([tuple['context'] for tuple in contexts['data'][:5]])
    x = vectorizer.fit_transform(corpus)
    x = x.toarray()
    # features = np.zeros((3, 2))
    # print(features)

    features = []
    for row_idx in range(x.shape[0]):
        features.append(utils.build_features(x[row_idx, :]))

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
