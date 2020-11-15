import json


def build_question_and_context(filename):
    """Build the question and context lists given a FQuAD file (train or validation).
    The chosen format is more adapted to work on the data providen.

    Parameters
    ----------
    filename : str
        The relative path to the data filename (JSON file on the FQuAD format)

    Returns
    -------
    tuple (of lists)
        Format of the lists returned:
        - for question list: [{'question': ..., 'question_id': ..., 'context_id': ...}, ...]
        - for the context list: [{'context': ..., 'context_id': ...}, ...]
    """
    with open(filename, encoding="utf8") as json_file:
        train = json.load(json_file)
    questions = []
    contexts = []

    for book_idx, book in enumerate(train['data']):
        for paragraph_idx, paragraph in enumerate(book['paragraphs']):
            for question_idx, qas in enumerate(paragraph['qas']):
                questions.append(
                    {
                        'question': qas['question'],
                        'question_id': f'{book_idx}_{paragraph_idx}_{question_idx}',
                        'context_id': f'{book_idx}_{paragraph_idx}',
                    }
                )
            contexts.append(
                {'context': paragraph['context'], 'context_id': f'{book_idx}_{paragraph_idx}'}
            )

    return questions, contexts
