from dataset.dataset import load_data, load_data_aug
from models.basic_model import get_model, get_model_aug
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from utils.utils import DataCollatorForMultipleChoice, compute_metrics, get_input_feature, compute_rouges
from transformers import set_seed
from transformers import AutoTokenizer, T5Tokenizer
import random
from tqdm import trange
import torch
import logging
from datetime import datetime

logging.basicConfig(filename='{:%Y-%m-%d %H:%M:%S}.log'.format(datetime.now()), filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
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


@torch.no_grad()
def eval(model, test_examples, tokenizer, eval_batch_size, choice_num, max_len, max_len_gen, device):
    count, count_right = 0, 0
    results = []
    model.eval()
    step_count = len(test_examples) // eval_batch_size
    if step_count * eval_batch_size < len(test_examples):
        step_count += 1
    step_trange = trange(step_count)
    sources, targets = [], []
    for step in step_trange:
        beg_index = step * eval_batch_size
        end_index = min((step + 1) * eval_batch_size, len(test_examples))
        batch_example = [example for example in test_examples[beg_index:end_index]]
        q_ids, q_mask, qo_ids, qo_mask, clue_ids, answers, output_clue = get_input_feature(batch_example,
                                                                                           max_len, max_len_gen,
                                                                                           device, tokenizer)
        scores, output_sequences = model(q_ids, q_mask, qo_ids, qo_mask, choice_num)

        scores = scores.cpu().detach().tolist()
        answers = answers.cpu().detach().tolist()
        p_anss = []
        for p, a, example in zip(scores, answers, batch_example):
            p_ans = p.index(max(p))
            p_anss.append(example['question']['choices'][p_ans]['label'])
            if p_ans == a:
                count_right += 1
            count += 1
        for sample, p_ans in zip(batch_example, p_anss):
            qid = sample['id']
            results.append(qid + "," + p_ans)
        predicts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        sources += predicts
        targets += output_clue

    rouge_score = compute_rouges(sources, targets)['rouge-l']

    return count_right / count, rouge_score, results

def aug_model(data_path_dict):
    device = "cpu"
    data = load_data_aug(data_path_dict)
    model = get_model_aug(device)
    train_examples = data['train']
    dev_examples = data['val']
    test_examples = data['test']
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    step_count, step_all, early_stop = 0, 0, 0
    tr_loss, nb_tr_steps = 0, 0
    gradient_accumulation_steps = 4
    train_batch_size, eval_batch_size = 16, 16
    lr = 5e-5
    max_source_length = 64
    max_len_gen = 32

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best_dev_acc, best_test_acc = 0, 0
    best_dev_rouge_score, best_test_rouge_score = 0, 0

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
                max_source_length=max_source_length,
                max_len_gen=max_len_gen,
                device=device,
                tokenizer=tokenizer)
            loss = model(q_ids, q_mask, qo_ids, qo_mask, clue_ids, answers)
            loss = loss.mean()
            tr_loss += loss.item()
            nb_tr_steps += 1
            loss = loss / gradient_accumulation_steps
            loss.backward()
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                # scheduler.step()
                optimizer.zero_grad()

            loss_show = ' Epoch:' + str(epoch) + " loss:" + str(round(tr_loss / nb_tr_steps, 4))
            step_trange.set_postfix_str(loss_show)

        dev_acc, dev_rouge_score, results_dev = eval(model, dev_examples, tokenizer, eval_batch_size,
                                                     4, max_source_length, max_len_gen, device)
        logging.info('dev_acc:' + str(dev_acc))
        if dev_acc > best_dev_acc:
            # save_dataset(path_save_result + '/dev.csv', results_dev)
            early_stop = 0
            test_acc, test_rouge_score, results_test = eval(model, test_examples, tokenizer, eval_batch_size,
                                                            4, max_source_length, max_len_gen, device)
            # save_dataset(path_save_result + '/test.csv', results_test)
            best_dev_acc, best_test_acc, best_dev_rouge_score, best_test_rouge_score = dev_acc, test_acc, dev_rouge_score, test_rouge_score

            # save_model(output_model_path, model, optimizer)
            logging.info("eval:%s", {'new best dev acc:': dev_acc, 'test_acc:': test_acc, 'rouge:': dev_rouge_score})

        if early_stop >= 5:
            break
    logging.info("eval:%s", {'best dev acc:': best_dev_acc, 'best_test_acc:': best_test_acc,
          'best_dev_rouge_score:': best_dev_rouge_score, 'best_test_rouge_score:': best_test_rouge_score})


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
