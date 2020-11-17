import argparse
from illuin.models import tfidf

MODELS = {'tfidf'}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Context finder CLI, based on Illuin's FQuAD dataset"
    )
    parser.add_argument(
        "-p",
        "--path",
        required=False,
        default="",
        type=str,
        help="path to JSON file (FQuAD format)",
    )
    parser.add_argument(
        "-m",
        "--model",
        required=False,
        default="tfidf",
        type=str,
        choices=MODELS,
        help="type of model chosen",
    )
    parser.add_argument(
        "-f",
        "--filemode",
        required=False,
        default="True",
        type=str,
        choices={"True", "False"},
        help="if the model is used for a file of questions or just a question",
    )
    parser.add_argument(
        "-n",
        required=False,
        default=-1,
        type=int,
        help="number of questions to get the context for (if file=True)",
    )
    parser.add_argument(
        "-r",
        "--random",
        required=False,
        default="False",
        type=str,
        choices={"True", "False"},
        help="randomization of the questions (if file=True)",
    )
    parser.add_argument(
        "-q",
        "--question",
        required=False,
        default="",
        type=str,
        help="question to get the context for (if file=False)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        required=False,
        default="False",
        type=str,
        choices={"True", "False"},
        help="whether to log some messages or not",
    )
    args = vars(parser.parse_args())
    filename = args["path"]
    filemode = args["filemode"] == "True"
    n = args["n"]
    random = args["random"] == "True"
    m = args["model"]
    question = args["question"]
    verbose = args["verbose"] == "True"
    if m == "tfidf":
        model = tfidf.TfidfClassifier(filename)
        _ = model.predict(file=filemode, n=n, random=random, question=question, verbose=verbose)
