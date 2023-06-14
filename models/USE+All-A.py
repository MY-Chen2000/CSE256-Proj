import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def baseline_model(data):
    correct = 0
    total = 0

    for item in data:
        choices = item['question']['choices']
        answer_key = item['answerKey']

        predicted_answer = choices[0]['label']

        if predicted_answer == answer_key:
            correct += 1
        total += 1

    accuracy = correct / total
    return accuracy

def get_data(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def get_accuracy(data):
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    correct = 0
    total = len(data)

    for example in data:
        question = example['question']['stem']
        choices = example['question']['choices']
        answer_key = example['answerKey']

        question_embedding = embed([question])[0]
        choice_embeddings = embed([choice['text'] for choice in choices])

        similarities = np.inner(question_embedding, choice_embeddings)

        predicted_index = np.argmax(similarities)
        predicted_answer = choices[predicted_index]['label']

        if predicted_answer == answer_key:
            correct += 1

    accuracy = correct / total
    return accuracy



file_path = 'test.jsonl'
test_data = load_jsonl(file_path)
All_A_accuracy = baseline_model(test_data)
print(f"Guess-All-Accuracy: {All_A_accuracy * 100:.2f}%")
USE_accuracy = get_accuracy(test_data)
print(f'USE-Accuracy: {USE_accuracy}')