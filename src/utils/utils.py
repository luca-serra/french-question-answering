import fr_core_news_sm
import string
import re

nlp = fr_core_news_sm.load()
PUNCTUATIONS = string.punctuation.replace("'", '').replace('[', '').replace(']', '')
PUNCTUATIONS = '[' + ' '.join(PUNCTUATIONS) + ' « »' + ']'
STOP_WORDS = nlp.Defaults.stop_words
ADDED_STOP_WORDS = ['t', 'y', "s'", "l'"]
for word in ADDED_STOP_WORDS:
    STOP_WORDS.add(word)
# tokenizer = nlp.Defaults.create_tokenizer(nlp)


def spacy_tokenizer(sentence):
    """Used to tokenize a sentence (with some preprocessing).
    Parameters
    ----------
    sentence : str
    Returns
    ----------
    List of the tokens in the sentence parameter.
    """
    tokens = tokenizer(sentence)
    # tokens = [word.lemma_ for word in tokens]
    tokens = [word.lower_ for word in tokens]
    # tokens = [
    #     word for word in tokens if not (str(word) in punctuations or str(word) in STOP_WORDS)
    # ]
    return tokens


# def custom_tokenizer(sentence, remove_duplicate=False):
#     tokens = ' '.join(re.sub(PUNCTUATIONS, ' ', sentence.lower()).split()).split(' ')
#     tokens_clean = []
#     for token in tokens:
#         if "'" in token:
#             split = token.split("'")
#             if split[0] != "":
#                 tokens_clean.append(split[0] + "'")
#             if split[1] != "":
#                 tokens_clean.append(split[1])
#         else:
#             tokens_clean.append(token)

#     if remove_duplicate:
#         tokens_clean = list(set(tokens_clean))

#     tokens_clean = [
#         word for word in tokens_clean if not (str(word) in STOP_WORDS)
#     ]
#     return tokens_clean

# def custom_stemmer(token):
#     endings = ['er', 'e', 'ant', 'ez', 'ent', 'é']
#     for ending in endings:
#         if token.endswith(ending):
#             return token[:-len(ending)]
#     return token


def custom_tokenizer(sentence, remove_duplicate=False):
    tokens = re.sub(PUNCTUATIONS, ' ', sentence.lower())
    tokens = tokens.replace('-', ' ').split()
    tokens_clean = []
    for token in tokens:
        # token = custom_stemmer(token)
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
    # tokens_clean = [
    #     word for word in tokens_clean if not (word in STOP_WORDS)
    # ]
    return tokens_clean


if __name__ == '__main__':
    # res = spacy_tokenizer(
    #     "Ma phrase de test doit être testée (et non pas tester) comme ceci, j'attends le résultat. Chanteur chanter chant ! Petit test; OK. L'apostrophe"
    # )
    # print(res)
    # context1 = "L'idée selon laquelle une planète inconnue pourrait exister entre les orbites de Mars et Jupiter fut proposée pour la première fois par Johann Elert Bode en 1768. Ses suggestions étaient basées sur la loi de Titius-Bode, une théorie désormais obsolète proposée par Johann Daniel Titius en 1766,. Selon cette loi, le demi-grand axe de cette planète aurait été d'environ 2,8 ua. La découverte d'Uranus par William Herschel en 1781 accrut la confiance dans la loi de Titius-Bode et, en 1800, vingt-quatre astronomes expérimentés combinèrent leurs efforts et entreprirent une recherche méthodique de la planète proposée,. Le groupe était dirigé par Franz Xaver von Zach. Bien qu'ils n'aient pas découvert Cérès, ils trouvèrent néanmoins plusieurs autres astéroïdes."
    # context2 = "Piazzi observa Cérès 24 fois, la dernière fois le 11 février. Le 24 janvier 1801, Piazzi annonça sa découverte par des lettres à plusieurs collègues italiens, parmi lesquels Barnaba Oriani à Milan. Il la décrivit comme une comète, mais remarqua que « puisque son mouvement est lent et uniforme, il m'a semblé à plusieurs reprises qu'il pourrait s'agir de quelque chose de mieux qu'une comète. » En avril, Piazzi envoya ses observations complètes à Oriani, Bode et Lalande à Paris. Elles furent publiées dans l'édition de septembre 1801 du Monatliche Correspondenz."
    # context3 = "Peu après sa découverte, Cérès s'approcha trop près du Soleil et ne put être observée à nouveau ; les autres astronomes ne purent confirmer les observations de Piazzi avant la fin de l'année. Cependant, après une telle durée, il était difficile de prédire la position exacte de Cérès. Afin de retrouver l'astéroïde, Carl Friedrich Gauss développa une méthode de déduction de l'orbite basée sur trois observations. En l'espace de quelques semaines, il prédit celle de Cérès et communiqua ses résultats à Franz Xaver von Zach, éditeur du Monatliche Correspondenz. Le 31 décembre 1801, von Zach et Heinrich Olbers confirmèrent que Cérès avait été retrouvée près de la position prévue, validant ainsi la méthode."
    # question1_1 = "Quel astronome a émit l'idée en premier d'une planète entre les orbites de Mars et Jupiter ?"
    # corpus = [question1_1, context1, context2, context3]
    # # print(spacy_tokenizer('y a-t-il'))
    # # vocabulary = spacy_tokenizer(question1_1)
    # # print(compute_tfidf_matrix(corpus, vocabulary, tokenizer))
    # print(corpus)
    # tokenizer = nlp.Defaults.create_tokenizer(nlp)
    # tokenized_corpus = []
    # for doc in tokenizer.pipe(corpus, batch_size=50):
    #     tokenized_corpus.append([doc])
    # print(tokenized_corpus)
    print(PUNCTUATIONS)
