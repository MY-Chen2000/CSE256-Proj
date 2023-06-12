import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_MCQ_Model(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(LSTM_MCQ_Model, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # self.vocab_size = vocab_size
        
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    
    def forward(self, question_embedded, choices_embedded):
        # question_embedded = self.embedding(question)
        # choices_embedded = self.embedding(choices).squeeze(0)
        
        _, (question_lstm_out, _) = self.lstm(question_embedded)
        choices_lstm_out = []
        for i in range(4):
            _, (choice_lstm_out, _) = self.lstm(choices_embedded[:,i,:,:])
            choices_lstm_out.append(choice_lstm_out)

        choices_lstm_out = torch.cat(choices_lstm_out, dim=0)
        
        logits = torch.matmul(choices_lstm_out.transpose(0,1), question_lstm_out.transpose(0,1).transpose(1,2)).squeeze()
        return nn.functional.softmax(logits, dim=0)

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attn_weights = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        # inputs: (batch_size, seq_length, hidden_size)

        # Compute attention scores
        attn_scores = self.attn_weights(inputs)
        attn_weights = self.softmax(attn_scores)

        # Apply attention weights to inputs
        attn_applied = torch.bmm(attn_weights, inputs)

        return attn_applied, attn_weights

class AttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AttentionModel, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.attention = AttentionLayer(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        # inputs: (batch_size, seq_length)

        embedded = self.embedding(inputs)
        # embedded: (batch_size, seq_length, hidden_size)

        output, hidden = self.gru(embedded)
        # output: (batch_size, seq_length, hidden_size)

        attn_output, attn_weights = self.attention(output)
        # attn_output: (batch_size, hidden_size)
        # attn_weights: (batch_size, seq_length)

        output = self.fc(attn_output)
        # output: (batch_size, output_size)

        return output, attn_weights

class LSTM_with_Attention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, use_bidirectional=False, use_dropout=False):
        super().__init__()
        self.rnn = nn.LSTM(embedding_dim, hidden_dim // 2,
                           bidirectional=use_bidirectional,
                           dropout=0.5 if use_dropout else 0.)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5 if use_dropout else 0.)

    def attention(self, lstm_output, final_state):
        # lstm_output = lstm_output.permute(1, 0, 2)
        merged_state = torch.cat(final_state, 1)
        # merged_state = merged_state.squeeze(0).unsqueeze(2)
        weights = torch.bmm(lstm_output, merged_state)
        weights = F.softmax(weights.squeeze(2), dim=1).unsqueeze(2)
        return torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2)

    def forward(self, question_embedded, choices_embedded):
        output, (hidden, _) = self.rnn(question_embedded)
        print(output.shape)
        attn_output = self.attention(output, hidden)
        print(attn_output.shape)
        question_repr = self.fc(attn_output.squeeze(0))

        choices_repr = []
        for choice_emb in choices_embedded:
            output, (hidden, _) = self.rnn(choice_emb)
            attn_output = self.attention(output, hidden)
            choices_repr.append(self.fc(attn_output.squeeze(0)))
        choices_repr = torch.cat(choices_repr, dim=0)

        logits = torch.matmul(choices_repr.transpose(0,1), question_repr.permute(1,2,0)).squeeze()
        return nn.functional.softmax(logits, dim=0)

class Pure_Attention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim,
                 n_layers=1, use_bidirectional=False, use_dropout=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim // 2,
                           bidirectional=use_bidirectional,
                           dropout=0.5 if use_dropout else 0.)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5 if use_dropout else 0.)

    def attention_net(self, lstm_output, final_state):
        lstm_output = lstm_output.permute(1, 0, 2)
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, dim=1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2),
                                     soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def attention(self, lstm_output, final_state):
        lstm_output = lstm_output.permute(1, 0, 2)
        merged_state = torch.cat([s for s in final_state], 1)
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        weights = torch.bmm(lstm_output, merged_state)
        weights = F.softmax(weights.squeeze(2), dim=1).unsqueeze(2)
        return torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.rnn(embedded)

        # attn_output = self.attention_net(output, hidden)
        attn_output = self.attention(output, hidden)

        return self.fc(attn_output.squeeze(0))