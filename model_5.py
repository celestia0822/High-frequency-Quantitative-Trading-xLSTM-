import torch.nn as nn
import torch
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        return self.fc(x)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.fc(x)

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.gru(x)
        x = x[:, -1, :]
        return self.fc(x)


class CNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim[0], hidden_dim[1], kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.seq_len = seq_len
        self.fc = nn.Linear(hidden_dim[1] * (self.seq_len // 4), output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class MLPModel(nn.Module):
    def __init__(self, input_dim: object, hidden_dim: object, output_dim: object) -> object:
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim[0])
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc3 = nn.Linear(hidden_dim[1], output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)