# Illuin: Find best context with FQuAD

In 2019, [Illuin](https://www.illuin.tech/) released the first French Question Answering Dataset (named [FQuAD](https://fquad.illuin.tech/)). A first (but crucial) step for Question Answering problems is to select the *context* in which the model will apply.

This project aims to develop a model able to give the most relevant context to work on given a question. This project has not for purpose to find the answer to the question (no span labelling).

## Requirements

You can install the required packages by running `pip install -r requirements.txt`.

## Data

The datasets used in this project are located in the `datasets/data` folder. To have them, follow the instructions of the `datasets` folder.