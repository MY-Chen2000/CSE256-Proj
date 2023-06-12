import spacy
nlp = spacy.load('en_core_web_md')

import numpy as np
import json
from scipy.spatial.distance import cosine
import heapq
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def get_token_embeddings(sentence):
    embs = []
    for token in nlp(sentence):
        if token.pos_ in ['NOUN', 'PRON', 'ADJ', 'VERB', 'PROPN', 'ADV']:
            embs.append(token.vector)
    if len(embs) == 0:
        embs.append(nlp("unknown").vector)
    return embs

def get_common_sense(question_with_fact_emb, common_knowledge, top_k):
    similarity_score = []
    for i, cs_emb in enumerate(common_knowledge):
        similarity_score.append((similarity_calculation(question_with_fact_emb, cs_emb), i))
    top_facts = heapq.nlargest(top_k, similarity_score)
    res = []
    for q in question_with_fact_emb:
        res.append(q)
    for _, i in top_facts:
        for j in range(len(common_knowledge[i])):
            res.append(common_knowledge[i][j])
    return res

def similarity_calculation(emb1, emb2):
    # return np.mean(cosine_similarity(emb1, emb2).reshape(-1))
    
    e1 = np.mean(emb1, axis=0)
    e2 = np.mean(emb2, axis=0)
    return cosine(e1, e2)

def load_data(file_path, common_knowledge, top_k=3):
    with open(file_path, "r") as json_file:
        json_list = list(json_file)

    X_no_facts = []
    X_with_facts = []
    X_with_facts_cs = []
    y = []

    for json_str in json_list:
        result = json.loads(json_str)
        fact = result["fact1"]
        question = result["question"]["stem"]

        q_no_fact_emb = get_token_embeddings(question)
        q_with_fact_emb = get_token_embeddings(fact + ' ' + question)

        q_with_fact_with_cs = get_common_sense(q_with_fact_emb, common_knowledge, top_k)
        
        choices_no_fact = [None] * 4
        choices_with_fact = [None] * 4
        choices_with_fact_cs = [None] * 4
        for choice in result["question"]["choices"]:
            choice_emb = get_token_embeddings(choice["text"])
            choices_no_fact[ord(choice["label"])-ord('A')] = similarity_calculation(q_no_fact_emb, choice_emb)
            choices_with_fact[ord(choice["label"])-ord('A')] = similarity_calculation(q_with_fact_emb, choice_emb)
            choices_with_fact_cs[ord(choice["label"])-ord('A')] = similarity_calculation(q_with_fact_with_cs, choice_emb)
        answer = ord(result["answerKey"]) - ord('A')

        X_no_facts.append(choices_no_fact)
        X_with_facts.append(choices_with_fact)
        X_with_facts_cs.append(choices_with_fact_cs)
        y.append(answer)

    return X_no_facts, X_with_facts, X_with_facts_cs, y

def load_common_knowledge(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    res = []
    for line in lines:
        if len(line) == 0:
            continue
        embs = []
        for token in nlp(line):
            if token.pos_ in ['NOUN', 'PRON', 'ADJ', 'VERB', 'PROPN', 'ADV']:
                embs.append(np.array(token.vector))
        if len(embs) == 0:
            continue
        res.append(np.array(embs))
    print(f"{len(res)} common knowledge loaded")
    return res

common_knowledge = load_common_knowledge("../Data/Additional/crowdsourced-facts.txt")
X_train_no_fact, X_train_with_fact, X_train_with_fact_cs, y_train = load_data("../Data/Additional/train_complete.jsonl", common_knowledge)
print('train loaded')
X_test_no_fact, X_test_with_fact, X_test_with_fact_cs, y_test = load_data("../Data/Additional/test_complete.jsonl", common_knowledge)
print('test loaded')

lin_clf = LinearSVC(random_state=2, tol=1e-5)
lin_clf.fit(X_train_no_fact, y_train)
y_pred = lin_clf.predict(X_test_no_fact)
print(accuracy_score(y_true=y_test, y_pred=y_pred))

lin_clf = LinearSVC(random_state=2, tol=1e-5)
lin_clf.fit(X_train_with_fact, y_train)
y_pred = lin_clf.predict(X_test_with_fact)
print(accuracy_score(y_true=y_test, y_pred=y_pred))

lin_clf = LinearSVC(random_state=2, tol=1e-5)
lin_clf.fit(X_train_with_fact_cs, y_train)
y_pred = lin_clf.predict(X_test_with_fact_cs)
print(accuracy_score(y_true=y_test, y_pred=y_pred))