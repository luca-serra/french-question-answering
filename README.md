# Illuin: Find the best context for Question Answering (QA)

In 2019, [Illuin](https://www.illuin.tech/) released the first French Question Answering Dataset (named [FQuAD](https://fquad.illuin.tech/)). A first (but important) step for QA problems is to select the *context* in which the model will apply.

This project aims to develop a model able to give the most relevant context to work on given a question. (This project has not for purpose to find the answer to the question.)

## Requirements

You can install the required packages by running `pip install -r requirements.txt`. Since the code uses absolute imports from the root of the project, you need to install the project by running `pip install -e path/to/project`. To install `fr_core_news_sm`, run `python -m spacy download fr_core_news_sm`.

## Data *(optional)*

The project is based on a split of FQuAD dataset (into a question file and a context file). To have them, follow the instructions of the `datasets` folder. 

*NB: installing these datasets is optional. It is not required to run the model. It is only suited if one wants to see how the data is used in the model.*

## Usage of the Command Line Interface (CLI)

Before using the CLI, download the train.json/valid.json files from the [FQuAD website](https://fquad.illuin.tech/) and place them into the `datasets/data` folder.
To run the model, one can execute the command:
`cd illuin && python main.py [options]`
The possible `options` are the following:
- `-p`/`--path` path_to_file: path to JSON file (FQuAD format)
- `-m`/`--model` model: name of the model to use
- `-n` number_of_questions: the number of questions to be answered
- `-r`/`--random` boolean: whether the questions are in the file order
- `-v`/`--verbose` boolean: whether to log some messages during prediction

The CLI can be used with a question asked by the user. For this, one may use:

- `-f`/`--filemode` False (use the model with one question defined in the CLI)
- `-q`/`--question` question: the question asked by the user

All these options are optional (except `-q`/`--question` if `-f`/`--filemode` False) and the default options are:

`-m` tfidf `-f` True `-n` -1 `-r` False  `-v` False

Examples:

The performances described in the table below were obtained with the following commands:
`python main.py -p ../datasets/data/train.json -n 1000 -r True -v True`

and

`python main.py -p ../datasets/data/valid.json -v True`

## Tests

In the `tests` folder, there are unit tests for the custom tokenizer used in this project. You can run the tests using: `python -m pytest`.

## Performance

The algorithm performances are the following:

|                              | accuracy      |  accuracy top 5 | prediction time (s)
| ----------------------------| ------------- |------------- | ------------- |
| on train set (1000 questions)  | 0.57      | 0.77         | 0.6 |
| on valid set (3188 questions)| 0.62        | 0.83         | 0.1 |

----

## Conclusion: 

The TFIDF approach has shown some satisfying results but may certainly be improved. Indeed, it does not account for the order of the words in the documents for instance. 

A word embedding approach has been tested (see `embedded_draft.py`) but due to lack of time, no conclusion can be made about its performances for the moment.
