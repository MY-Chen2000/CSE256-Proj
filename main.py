from dataset.dataset import load_data, load_data_aug
from models.basic_model import get_model, get_model_aug
from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer
from utils.utils import DataCollatorForMultipleChoice, compute_metrics, get_input_feature
from transformers import set_seed
import random
from tqdm import trange
import torch
torch.manual_seed(0)
set_seed(0)


def base_model(data_path_dict):
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    encoded_datasets = load_data(data_path_dict, tokenizer)

    model = get_model(model_name)
    args = TrainingArguments(f"{model_name}-finetuned-swag",
                             evaluation_strategy="epoch",
                             learning_rate=5e-5,
                             per_device_train_batch_size=16,
                             num_train_epochs=3,
                             load_best_model_at_end=True,
                             save_strategy='epoch',
                             weight_decay=0.01)
    accepted_keys = ["input_ids", "attention_mask", "label"]
    features = [{k: v for k, v in encoded_datasets["train"][i].items() if k in accepted_keys} for i in range(10)]
    batch = DataCollatorForMultipleChoice(tokenizer)(features)

    [tokenizer.decode(batch["input_ids"][8][i].tolist()) for i in range(4)]

    trainer = Trainer(model,
                      args,
                      train_dataset=encoded_datasets["train"],
                      eval_dataset=encoded_datasets["validation"],
                      tokenizer=tokenizer,
                      data_collator=DataCollatorForMultipleChoice(tokenizer),
                      compute_metrics=compute_metrics)

    trainer.train()
    print('#########')
    print('test set:')
    print('#########')
    final_eval = trainer.evaluate(eval_dataset=encoded_datasets['test'])
    print(final_eval)


def aug_model(data_path_dict):
    data = load_data_aug(data_path_dict)
    model = get_model_aug("cpu")
    train_examples = data['train']
    val_examples = data['val']
    test_examples = data['test']

    step_count, step_all, early_stop = 0, 0, 0

    train_batch_size = 2
    for epoch in range(10):
        early_stop += 1
        order = list(range(len(train_examples)))
        random.seed(0 + epoch)
        random.shuffle(order)

        step_count = len(train_examples) // train_batch_size
        if step_count * train_batch_size < len(train_examples):
            step_count += 1
        step_trange = trange(step_count)
        for step in step_trange:
            step_all += 1
            beg_index = step * train_batch_size
            end_index = min((step + 1) * train_batch_size, len(train_examples))
            order_index = order[beg_index:end_index]
            batch_example = [train_examples[index] for index in order_index]
            q_ids, q_mask, qo_ids, qo_mask, clue_ids, answers, output_clue = get_input_feature(
                batch_example,
                max_source_length=64,
                max_len_gen=32,
                device='cpu')
            loss = model(q_ids, q_mask, qo_ids, qo_mask, clue_ids, answers)

        if early_stop >= 5:
            break




if __name__ == '__main__':
    with_fact = {
        'train': 'Data/Additional/train_complete.jsonl',
        'val': 'Data/Additional/dev_complete.jsonl',
        'test': 'Data/Additional/test_complete.jsonl'
    }
    wo_fact = {
        'train': 'Data/Main/train.jsonl',
        'val': 'Data/Main/dev.jsonl',
        'test': 'Data/Main/test.jsonl'
    }
    aug_model(with_fact)
