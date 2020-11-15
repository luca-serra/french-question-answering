"""
This module uses the training dataset of FQuAD (available here https://fquad.illuin.tech/)
to create a new training set, more adapted to the given project.
Here, it is more appropriate to have a dataset containing directly the questions
and their context.
"""
import json
from illuin.utils import question_and_context

FQUAD_FILENAME = '../../datasets/data/train.json'

if __name__ == "__main__":
    questions, contexts = question_and_context.build_question_and_context(FQUAD_FILENAME)
    with open('../../datasets/data/questions.json', 'w', encoding="utf8") as outfile:
        json.dump({'data': questions, 'version': 1.0}, outfile, ensure_ascii=False, indent=4)

    with open('../../datasets/data/contexts.json', 'w', encoding="utf8") as outfile:
        json.dump({'data': contexts, 'version': 1.0}, outfile, ensure_ascii=False, indent=4)
