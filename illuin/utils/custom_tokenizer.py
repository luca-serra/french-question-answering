import fr_core_news_sm
import string
import re

nlp = fr_core_news_sm.load()
PUNCTUATIONS = string.punctuation.replace("'", '').replace('[', '').replace(']', '')
PUNCTUATIONS = '[' + ' '.join(PUNCTUATIONS) + ' « »' + ']'
STOP_WORDS = nlp.Defaults.stop_words
ADDED_STOP_WORDS = ['t', 'y', "s'", "l'", 's']  # stop words that I found to be useful to increase performance
for word in ADDED_STOP_WORDS:
    STOP_WORDS.add(word)


def custom_tokenizer(sentence, remove_duplicate=False):
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


if __name__ == '__main__':
    # Test
    context1 = "L'idée selon laquelle une planète inconnue pourrait exister entre les orbites de Mars et Jupiter fut proposée pour la première fois par Johann Elert Bode en 1768. Ses suggestions étaient basées sur la loi de Titius-Bode, une théorie désormais obsolète proposée par Johann Daniel Titius en 1766,. Selon cette loi, le demi-grand axe de cette planète aurait été d'environ 2,8 ua. La découverte d'Uranus par William Herschel en 1781 accrut la confiance dans la loi de Titius-Bode et, en 1800, vingt-quatre astronomes expérimentés combinèrent leurs efforts et entreprirent une recherche méthodique de la planète proposée,. Le groupe était dirigé par Franz Xaver von Zach. Bien qu'ils n'aient pas découvert Cérès, ils trouvèrent néanmoins plusieurs autres astéroïdes."
    print(custom_tokenizer(context1))
