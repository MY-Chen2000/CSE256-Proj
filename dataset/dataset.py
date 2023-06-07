from datasets import load_dataset, load_metric, ClassLabel
from pprint import pprint
from transformers import AutoTokenizer

ending_names = ['A', 'B', 'C', 'D']

def choices(example):
    for dic in example['question.choices']:
        example[dic['label']] = dic['text']
    example.pop('question.choices', None)
#    example.pop('question.stem', None)
    return example


def preprocess_function(examples, tokenizer):
    # Repeat each first sentence four times to go with the four possibilities of second sentences.
    first_sentences = [[context] * 4 for context in examples["fact1"]]
    # Grab all second sentences possible for each context.
    question_headers = examples["question.stem"]
    second_sentences = [[f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)]

    # Flatten everything
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # Tokenize
    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    # Un-flatten
    return {k: [v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}


def load_data(files, tokenizer):
    # files: {'train': path_to_train.jsonl, 'val': path_to_val.jsonl, 'test': path_to_test.jsonl, }
    openbookQA = load_dataset('json', data_files={'train': files['train'],
                                                  'validation': files['val'],
                                                  'test': files['test']})
    flatten = openbookQA.flatten()
    updated = flatten.map(choices)
    updated = updated.rename_column('answerKey', 'label')
    # pprint(updated['train'][0])

    encoded_datasets = updated.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)

    return encoded_datasets
