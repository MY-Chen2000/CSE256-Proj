import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import OpenBookqaDataset
from sklearn.metrics import accuracy_score
import pandas as pd
from model import LSTM_MCQ_Model, LSTM_with_Attention


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

embedding_dim = 300
hidden_dim = 200
output_dim = 100

# Create an instance of the LSTM model
# model = LSTM_MCQ_Model(embedding_dim, hidden_dim).to(device)
model = LSTM_with_Attention(embedding_dim, hidden_dim, output_dim).to(device)

# Define loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Prepare input data
train = OpenBookqaDataset('../Data/Additional/train.csv')
test = OpenBookqaDataset('../Data/Additional/test.csv')
train_loader = DataLoader(train, batch_size=64, shuffle=True)
test_loader = DataLoader(test, batch_size=1, shuffle=False)

# Training loop
num_epochs = 10
model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    running_loss = 0
    for question_only, question_with_fact, question_with_fact_cs, choices, answer in train_loader:
        logits = model(question_with_fact_cs.to(device), choices.to(device))
        loss = loss_function(logits, answer.float().to(device))
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")


with torch.no_grad():
    model.eval()
    y_pred = []
    y_true = []
    for question_only, question_with_fact, question_with_fact_cs, choices, answer in test_loader:
        logits = model(question_with_fact_cs.to(device), choices.to(device))
        y_pred.append(torch.argmax(logits).cpu().numpy())
        y_true.append(torch.argmax(answer).cpu().numpy())

    pd.DataFrame([y_pred, y_true]).to_csv('result.csv')
    print(accuracy_score(y_pred, y_true))