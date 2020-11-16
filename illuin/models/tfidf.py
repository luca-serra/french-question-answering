"""
Implementing a TFIDF approach to find the best context for a given question
"""
import time
import random as ra
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from illuin.utils import custom_tokenizer, utils, question_and_context


class TfidfClassifier(BaseEstimator):
    def __init__(self, filename):
        """filename is the relative path to the JSON file on the FQuAD dataset format"""
        self.filename = filename

    def fit(self, X, y=None):
        return self  # for TFIDF model, there is no fit step

    def predict(self, file=True, n=-1, random=False, question=None, verbose=False):
        if file:
            questions, contexts = question_and_context.build_question_and_context(self.filename)
            if n > 0:
                questions = questions[:n]
            if random:
                ra.shuffle(questions)
        else:
            _, contexts = question_and_context.build_question_and_context(self.filename)
            questions = [{'question': question, 'question_id': 0, 'context_id': None}]

        context_ids = [tuple['context_id'] for tuple in contexts]

        count_avg = 0  # counting the number of correct predictions based on the average of TDFIDF matrix (row)
        count_ratio = 0  # counting the number of correct predictions based on the ratio of non-zero terms in the row
        count_top5_avg = 0  # same but with a top5 prediction
        count_top5_ratio = 0  # same but with a top5 prediction
        for q in questions:
            start_time = time.time()
            question = q['question']
            context_id = q['context_id']
            corpus = [question] + [tuple['context'] for tuple in contexts]
            tokenizer = custom_tokenizer.custom_tokenizer
            vocabulary = tokenizer(question, remove_duplicate=True)
            vectorizer = TfidfVectorizer(
                vocabulary=vocabulary, tokenizer=tokenizer, norm=None, binary=True
            )
            tfidf_matrix = vectorizer.fit_transform(corpus)
            tfidf_matrix = tfidf_matrix.toarray()

            features = utils.build_features(tfidf_matrix)
            if verbose:
                print(f'Question: {question}')
                print(f'Vocabulary: {vocabulary}')
            indices_top5_avg = np.argpartition(features[1:, 0], -5)[-5:]
            indices_top5_ratio = np.argpartition(features[1:, 1], -5)[-5:]
            predicted_contexts_top5_avg = [contexts[idx]['context_id'] for idx in indices_top5_avg]
            predicted_contexts_top5_ratio = [
                contexts[idx]['context_id'] for idx in indices_top5_ratio
            ]
            predicted_context_avg = contexts[np.argmax(features[1:, 0])]['context_id']
            predicted_context_ratio = contexts[np.argmax(features[1:, 1])]['context_id']
            if context_id == predicted_context_avg:
                count_avg += 1
            if context_id == predicted_context_ratio:
                count_ratio += 1
            if context_id in predicted_contexts_top5_avg:
                if verbose:
                    print('✓')
                count_top5_avg += 1
            else:
                if verbose:
                    print(predicted_contexts_top5_avg)
                    if not (context_id is None):
                        print('✘')
                        print(context_id)
                        print(tfidf_matrix[1 + context_ids.index(context_id), :])
                    else:
                        print('↑ top 5 suggested contexts for the question')
            if context_id in predicted_contexts_top5_ratio:
                count_top5_ratio += 1
            if verbose:
                print(f'Answered in {time.time() - start_time:.2f} sec')

        if file:
            print(
                '----------------------------------------------------------------\n',
                f'Number of questions: {n}, accuracy_avg: {count_avg/n}, accuracy_ratio: {count_ratio/n},\
            accuracy top5 avg: {count_top5_avg/n}, accuracy top5 ratio: {count_top5_ratio/n}',
            )


# ['test de mon algorithme', 'ceci est un test à tester', "Ma phrase de test doit être testée (et non pas tester) comme ceci, j'attends le résultat. Chanteur chanter chant ! Petit test; OK. L'apostrophe"]
