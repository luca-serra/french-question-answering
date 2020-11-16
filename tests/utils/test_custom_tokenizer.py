import unittest
from collections import Counter
from illuin.utils import custom_tokenizer
from tests.utils import constants

PUNCTUATIONS = constants.PUNCTUATIONS
STOP_WORDS = constants.STOP_WORDS


class TestCustomTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_sentence = "Ceci est une phrase de test !\
            Une fois traitée, il ne doit pas y avoir de mots génériques comme «n'est-ce pas»,\
            de ponctuations comme «; ? / .», de « s », de doublon (doublon ?) et tout doit être\
            en minuscule."
        cls.tokenized_sentence = custom_tokenizer.custom_tokenizer(test_sentence)
        cls.tokenized_sentence_no_duplicate = custom_tokenizer.custom_tokenizer(
            test_sentence, remove_duplicate=True
        )

    def test_no_punctuation(self):
        """Assert there is no punctuation in tokenized sentence"""
        self.assertNotIn(PUNCTUATIONS, self.tokenized_sentence)

    def test_no_stop_words(self):
        """Assert there is no stop word in tokenized sentence"""
        self.assertNotIn(STOP_WORDS, self.tokenized_sentence)

    def test_no_s(self):
        """Assert there is no 's' letter at the end of each token in tokenized sentence"""
        self.assertNotIn('s', [token[-1] for token in self.tokenized_sentence])

    def test_no_duplicate(self):
        """Assert there is no duplicate in tokenized sentence if remove_duplicate=True"""
        token_occurences = list(Counter(self.tokenized_sentence_no_duplicate).values())
        self.assertTrue(all([occurence == 1 for occurence in token_occurences]))

    def test_lowercase(self):
        """Assert the tokenized sentence is in lower case"""
        self.assertTrue(all([token == token.lower() for token in self.tokenized_sentence]))