from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from dataclasses import dataclass
import numpy as np
from typing import Optional, Union
from transformers import AutoTokenizer, T5Tokenizer
import torch
import rouge

ending_names = ['A', 'B', 'C', 'D']

def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [ending_names.index(feature.pop(label_name)) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def get_input_feature(samples, max_source_length, max_len_gen, device, tokenizer):
    sep = ' '
    output_clue = []
    answers = []
    input_ids_q, attention_mask_q = [], []
    input_ids_qo, attention_mask_qo = [], []
    for sample in samples:
        answerKey = sample['answerKey']
        question = sample['question']['stem']
        content = sample['fact1']
        for o_i, (opt, opt_name) in enumerate(zip(sample['question']['choices'], 'ABCD')):
            option = opt['text']
            input_ids_qo.append(content + question + sep + option)


        input_ids_q.append(content + question + sep)
        answer = ord(answerKey) - ord('A')
        answers.append(answer)
        output_clue.append(sample['question']['choices'][answer]['text'])

    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    def tokenizer_fun(input_ids, max_len):
        encoding = tokenizer(input_ids,
                             padding='longest',
                             max_length=max_len,
                             truncation=True,
                             return_tensors="pt")
        ids = encoding.input_ids.to(device)
        mask = encoding.attention_mask.to(device)
        return ids, mask

    q_ids, q_mask = tokenizer_fun(input_ids_q, max_source_length)
    qo_ids, qo_mask = tokenizer_fun(input_ids_qo, max_source_length)
    clue_ids, _ = tokenizer_fun(output_clue, max_len_gen)
    clue_ids = torch.tensor(clue_ids, dtype=torch.long).to(device)
    answers = torch.tensor(answers, dtype=torch.long).to(device)
    return q_ids, q_mask, qo_ids, qo_mask, clue_ids, answers, output_clue



rouge = rouge.Rouge()
def compute_rouge(source, target):

    source, target = ' '.join(source), ' '.join(target)
    try:
        scores = rouge.get_scores(hyps=source, refs=target)
        return {
            'rouge-1': scores[0]['rouge-1']['f'],
            'rouge-2': scores[0]['rouge-2']['f'],
            'rouge-l': scores[0]['rouge-l']['f'],
        }
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }


def compute_rouges(sources, targets):
    scores = {
        'rouge-1': 0.0,
        'rouge-2': 0.0,
        'rouge-l': 0.0,
    }
    for source, target in zip(sources, targets):
        score = compute_rouge(source, target)
        for k, v in scores.items():
            scores[k] = v + score[k]
    return {k: v / len(targets) for k, v in scores.items()}

