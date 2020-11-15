import json

with open('../datasets/data/contexts.json', encoding="utf8") as json_file:
    contexts = json.load(json_file)

with open('../datasets/data/questions.json', encoding="utf8") as json_file:
    questions = json.load(json_file)

if __name__ == "__main__":