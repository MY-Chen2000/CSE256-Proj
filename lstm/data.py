import json
import os
# import tensorflow_hub as hub
# from sklearn.metrics import pairwise_distances
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import csv
import pandas as pd
import torch
from torch.utils.data import Dataset
import spacy

def get_common_sense(sentences_with_choices, common_knowledge, TOP_K):
    ps = PorterStemmer()

    processed_sentences_with_choices = []
    for sent in sentences_with_choices:
        processed_sentences_with_choices.append(' '.join([ps.stem(w) for w in word_tokenize(sent)]))

    processed_common_knowledge = []
    for sent in common_knowledge:
        processed_common_knowledge.append(' '.join([ps.stem(w) for w in word_tokenize(sent)]))

    num_of_questions = len(processed_sentences_with_choices)

    vectorizer = TfidfVectorizer()
    processed_common_knowledge.extend(processed_sentences_with_choices)
    
    X = vectorizer.fit_transform(processed_common_knowledge)
    cs = X[:-num_of_questions]
    q = X[-num_of_questions:]
    similarity_score = cosine_similarity(q, cs)
    top_results = np.argpartition(-similarity_score, range(TOP_K), axis=1)

    return top_results

def load_data(file_path, common_knowledge, top_k=2):
    with open(file_path, "r") as json_file:
        json_list = list(json_file)

    data = []
    sentences_with_choices = []

    for json_str in json_list:
        result = json.loads(json_str)
        record = {}
        record["fact"] = result["fact1"]
        record["question"] = result["question"]["stem"]
        for choice in result["question"]["choices"]:
            record[choice["label"]] = choice["text"]
        record["answer"] = result["answerKey"]
        data.append(record)
        sent_with_fact = (
            record["question"] + " " + record["fact"]
        )
        sentences_with_choices.append(sent_with_fact)

    top_related_facts = get_common_sense(sentences_with_choices, common_knowledge, top_k)
    for i in range(len(top_related_facts)):
        data[i]['commonSense'] = [common_knowledge[top_related_facts[i][j]] for j in range(top_k)]
    print(f"{len(data)} records loaded")
    return data


def load_common_knowledge(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    res = []
    for line in lines:
        if len(line) == 0:
            continue
        res.append(line.strip())
    print(f"{len(res)} common knowledge loaded")
    return res

def write_data(file_path, data):
    with open(file_path, 'w', newline='') as csvfile:
        field_names = ['fact', 'question', 'A', 'B', 'C', 'D', 'commonSense', 'answer']
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(data)

# def universal_sentence_encoder(sentences):
#     embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
#     embeddings = embed(sentences)
#     return embeddings

class OpenBookqaDataset(Dataset):
    def __init__(self, file_path):
        self.question_list = pd.read_csv(file_path)
        self.nlp = spacy.load('en_core_web_md')

    def __len__(self):
        return len(self.question_list)

    def __getitem__(self, idx):
        question = self.question_list.iloc[idx]['question']
        fact = self.question_list.iloc[idx]['fact']
        common_sense = self.question_list.iloc[idx]['commonSense']

        question_only = self.__get_token(question, 25)
        question_with_fact = self.__get_token(question + ' ' + fact, 35)
        question_with_fact_cs = self.__get_token(question + ' ' + fact + ' ' + ' '.join(common_sense), 100)

        choices = []
        for choice in ['A', 'B', 'C', 'D']:
            choices.append(self.__get_token(self.question_list.iloc[idx][choice], 5).unsqueeze(0))
        choices = torch.cat(choices, dim=0)
        
        answer = [0] * 4
        answer[ord(self.question_list.iloc[idx]['answer'])-ord('A')] = 1
        answer = torch.tensor(answer)

        return question_only, question_with_fact, question_with_fact_cs, choices, answer
    
    def __get_token(self, sentence, length):
        embs = []
        for token in self.nlp(sentence):
            if token.pos_ in ['NOUN', 'PRON', 'ADJ', 'VERB', 'PROPN', 'ADV']:
                embs.append(torch.tensor(token.vector).reshape(1, -1))
        if len(embs) == 0:
            embs.append(torch.tensor(self.nlp("UNK").vector).reshape(1, -1))
        if len(embs) > length:
            embs = embs[:length]
        else:
            for _ in range(length-len(embs)):
                embs.append(torch.tensor(np.zeros(300)).reshape(1, -1).float())
        return torch.cat(embs, dim=0)


if __name__ == "__main__":
    data_root_path = "../Data/Additional"

    common_knowledge = load_common_knowledge(os.path.join(data_root_path, "crowdsourced-facts.txt"))
    # common_knwoledge_emb = universal_sentence_encoder(common_knowledge)
    # common_knowledge_with_emb = (common_knowledge, common_knwoledge_emb)

    train_data = load_data(os.path.join(data_root_path, "train_complete.jsonl"), common_knowledge)
    valid_data = load_data(os.path.join(data_root_path, "dev_complete.jsonl"), common_knowledge)
    test_data = load_data(os.path.join(data_root_path, "test_complete.jsonl"), common_knowledge)

    train_output_path = (os.path.join(data_root_path, "train.csv"))
    valid_output_path = (os.path.join(data_root_path, "valid.csv"))
    test_output_path = (os.path.join(data_root_path, "test.csv"))

    write_data(train_output_path, train_data)
    write_data(valid_output_path, valid_data)
    write_data(test_output_path, test_data)
