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
        """Return an array of predicted contexts given the 'filename' used at instance creation

        Parameters
        ----------
        file : bool, optional
            Whether to run prediction on a JSON file or a str question, by default True
        n : int, optional
            Number of questions to answer (if file), by default -1
        random : bool, optional
            Whether to randomize the questions (if file), by default False
        question : str, optional
            The question to be answered (if file=False), by default None
        verbose : bool, optional
            Whether to log messages during prediction, by default False

        Returns
        -------
        array
            The predicted contexts
        """
        if file:
            questions, contexts = question_and_context.build_question_and_context(self.filename)
            if random:
                ra.shuffle(questions)
            if n > 0:
                questions = questions[:n]
            else:
                n = len(questions)
        else:
            _, contexts = question_and_context.build_question_and_context(self.filename)
            questions = [{'question': question, 'question_id': 0, 'context_id': None}]

        context_ids = [tuple['context_id'] for tuple in contexts]

        count_avg = 0  # counting the number of correct predictions based on the average of TDFIDF matrix (row)
        count_ratio = 0  # counting the number of correct predictions based on the ratio of non-zero terms in the row
        count_top5_avg = 0  # same but with a top5 prediction
        count_top5_ratio = 0  # same but with a top5 prediction
        predictions_avg = []  # the predictions (based on average) which will be returned
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
                print('-----------------------------------------')
                print(f'Question: {question}')
                print(f'Vocabulary: {vocabulary}')
            indices_top5_avg = utils.argmax_n(features[1:, 0], 5)
            indices_top5_ratio = utils.argmax_n(features[1:, 1], 5)
            predicted_contexts_top5_avg = [contexts[idx]['context_id'] for idx in indices_top5_avg]
            predicted_contexts_top5_ratio = [
                contexts[idx]['context_id'] for idx in indices_top5_ratio
            ]
            predicted_context_avg = predicted_contexts_top5_avg[0]
            predictions_avg.append(predicted_context_avg)
            predicted_context_ratio = predicted_contexts_top5_ratio[0]
            if context_id == predicted_context_avg:
                count_avg += 1
            if context_id == predicted_context_ratio:
                count_ratio += 1
            if context_id in predicted_contexts_top5_avg:
                if verbose:
                    print('✓')
                    print(f'Top 5 predicted contexts: {predicted_contexts_top5_avg}')
                    print(f'(True context: {context_id})')
                count_top5_avg += 1
            else:
                if verbose:
                    print(f'Top 5 predicted contexts: {predicted_contexts_top5_avg}')
                    if not (context_id is None):
                        print('✘')
                        print(f'True context: {context_id}')
                        print(
                            f'TFIDF row for the true context: {tfidf_matrix[1 + context_ids.index(context_id), :]}'
                        )
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
        return np.array(predictions_avg)
