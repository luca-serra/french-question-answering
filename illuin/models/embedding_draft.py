"""
This module is not finished. This is a draft to test word embedding approach
on the given problem.
"""

import fr_core_news_sm
import numpy as np
import string

from illuin.utils import custom_tokenizer, utils, question_and_context

nlp = fr_core_news_sm.load()
STOP_WORDS = nlp.Defaults.stop_words
PUNCTUATIONS = string.punctuation


questions, contexts = question_and_context.build_question_and_context(
    '../../datasets/data/train.json'
)

corpus = [tuple['context'] for tuple in contexts]  # corpus of all contexts
corpus_tokenized = []


def tokenizer(sentence):
    sentence = ' '.join(custom_tokenizer.custom_tokenizer(sentence))
    tokens = nlp(sentence)
    return tokens


n = len(corpus)
for i, sentence in enumerate(corpus):
    print('\r', f'{100 * i / n:.1f}%', end='')
    corpus_tokenized.append(tokenizer(sentence))

question_number_to_be_tested = 5
question = questions[question_number_to_be_tested]['question']


question_tokenized = tokenizer(question)
question_vectorized = np.mean(
    [token.vector if token.has_vector else np.zeros((96,)) for token in question_tokenized], axis=0
)
question_vectorized /= np.linalg.norm(question_vectorized)

scores = []

for document in corpus_tokenized:
    avg_vector = np.mean(
        [token.vector if token.has_vector else np.zeros((96,)) for token in document], axis=0
    )
    avg_vector /= np.linalg.norm(avg_vector)
    cos_similarity = np.dot(avg_vector, question_vectorized)
    scores.append(cos_similarity)
scores = np.array(scores)

indices_top5 = utils.argmax_n(scores, 5)
print(f"Top 5 predicted contexts: {[contexts[idx]['context_id'] for idx in indices_top5]}")
print(f"Actual context: {questions[question_number_to_be_tested]['context_id']}")