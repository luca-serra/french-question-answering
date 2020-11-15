"""
This module uses the training dataset of FQuAD (available here https://fquad.illuin.tech/)
to create a new training set, more adapted to the given project.
Here, it is more appropriate to have a dataset containing directly the questions
and their context.
"""
import json

with open('data/train.json', encoding="utf8") as json_file:
    train = json.load(json_file)  # download the training dataset and place it in the data folder
questions = (
    []
)  # the dataset which will be used by the model in this project. It links questions with context identifiers
contexts = []  # linking the contexts with identifiers

for book_idx, book in enumerate(train['data']):
    for paragraph_idx, paragraph in enumerate(book['paragraphs']):
        for qas in paragraph['qas']:
            questions.append({'question': qas['question'], 'context_id': f'{book_idx}_{paragraph_idx}'})
        contexts.append({'context': paragraph['context'], 'context_id': f'{book_idx}_{paragraph_idx}'})

with open('data/questions.json', 'w', encoding="utf8") as outfile:
    json.dump({'data': questions, 'version': 1.0}, outfile, ensure_ascii=False, indent=4)

with open('data/contexts.json', 'w', encoding="utf8") as outfile:
    json.dump({'data': contexts, 'version': 1.0}, outfile, ensure_ascii=False, indent=4)
