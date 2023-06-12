import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from scipy.spatial.distance import cosine

def load_glove_embeddings():
    glove_embeddings = {}
    filename = '../glove_embeddings/glove.42B.300d.txt'
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.readlines()
        for line in content:
            content_line_split = line.split()
            token = content_line_split[0]
            embedding = content_line_split[1:]
            embedding = np.array(embedding, dtype=np.float64)
            glove_embeddings[token] = embedding
    return glove_embeddings

def get_token_embeddings(sentence, glove_embeddings):
    ps = PorterStemmer()
    token_seq = [ps.stem(w.lower()) for w in word_tokenize(sentence)]
    embedding = np.zeros((len(token_seq), 300))
    for idx, token in enumerate(token_seq):
        if token in glove_embeddings:
            embedding[idx] = glove_embeddings[token]
        else:
            embedding[idx] = glove_embeddings['unk']
    return embedding

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

def similarity_calculation(emb1, emb2):
    return np.mean(cosine_similarity(emb1, emb2).reshape(-1))
    
    e1 = np.mean(emb1, axis=0)
    e2 = np.mean(emb2, axis=0)
    return cosine(e1, e2)

# def load_data(file_path, glove_embeddings, common_knowledge, top_k):
def load_data(file_path, glove_embeddings):
    with open(file_path, "r") as json_file:
        json_list = list(json_file)

    X_no_facts = []
    X_with_facts = []
    # X_with_facts_cs = []
    y = []

    for json_str in json_list:
        result = json.loads(json_str)
        fact = result["fact1"]
        question = result["question"]["stem"]

        # top_related_facts = get_common_sense(fact+question, common_knowledge, top_k)
        # related_fact = ' '.join([common_knowledge[top_related_facts[i][j]] for i in range(len(top_related_facts)) for j in range(top_k)])

        q_no_fact_emb = get_token_embeddings(question, glove_embeddings)
        q_with_fact_emb = get_token_embeddings(fact+question, glove_embeddings)
        # q_with_fact_with_cs = get_token_embeddings(fact+question+related_fact, glove_embeddings)

        choices_no_fact = [None] * 4
        choices_with_fact = [None] * 4
        # choices_with_fact_cs = [None] * 4
        for choice in result["question"]["choices"]:
            choice_emb = get_token_embeddings(choice["text"], glove_embeddings)
            choices_no_fact[ord(choice["label"])-ord('A')] = similarity_calculation(q_no_fact_emb, choice_emb)
            choices_with_fact[ord(choice["label"])-ord('A')] = similarity_calculation(q_with_fact_emb, choice_emb)
            # choices_with_fact_cs[ord(choice["label"])-ord('A')] = cosine(q_with_fact_with_cs, choice_emb)
        answer = ord(result["answerKey"]) - ord('A')

        X_no_facts.append(choices_no_fact)
        X_with_facts.append(choices_with_fact)
        # X_with_facts_cs.append(choices_with_fact_cs)
        y.append(answer)

    return X_no_facts, X_with_facts, y
    # return X_no_facts, X_with_facts, X_with_facts_cs, y

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

if __name__ == '__main__':
    glove_emb = load_glove_embeddings()
    # common_knowledge = load_common_knowledge("../Data/Additional/crowdsourced-facts.txt")

    X_train_no_fact, X_train_with_fact, y_train = load_data("../Data/Additional/train_complete.jsonl", glove_emb)
    # X_valid_no_fact, X_valid_with_fact, y_valid = load_data("../Data/Additional/dev_complete.jsonl", glove_emb)
    X_test_no_fact, X_test_with_fact, y_test = load_data("../Data/Additional/test_complete.jsonl", glove_emb)

    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score

    lin_clf = LinearSVC(random_state=2, tol=1e-4)
    lin_clf.fit(X_train_no_fact, y_train)
    y_pred = lin_clf.predict(X_test_no_fact)
    print(accuracy_score(y_true=y_test, y_pred=y_pred))

    lin_clf = LinearSVC(random_state=2, tol=1e-5)
    lin_clf.fit(X_train_with_fact, y_train)
    y_pred = lin_clf.predict(X_test_with_fact)
    print(accuracy_score(y_true=y_test, y_pred=y_pred))