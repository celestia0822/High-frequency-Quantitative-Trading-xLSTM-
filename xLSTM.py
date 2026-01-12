import torch
import torch.nn as nn
import math
from early_stopping import EarlyStopping
from dynamic_dropout import DynamicDropout

import csv
import time
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
# Define the mLSTM module
class mLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, mem_dim):
        super(mLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mem_dim = mem_dim
        self.w_q = nn.Parameter(torch.randn(input_size, hidden_size))
        self.b_q = nn.Parameter(torch.randn(hidden_size))
        self.w_k = nn.Parameter(torch.randn(input_size, mem_dim))
        self.b_k = nn.Parameter(torch.randn(mem_dim))
        self.w_v = nn.Parameter(torch.randn(input_size, mem_dim))
        self.b_v = nn.Parameter(torch.randn(mem_dim))
        self.w_i = nn.Parameter(torch.randn(input_size, 1))
        self.b_i = nn.Parameter(torch.randn(1))
        self.w_f = nn.Parameter(torch.randn(input_size, 1))
        self.b_f = nn.Parameter(torch.randn(1))
        self.w_o = nn.Parameter(torch.randn(input_size, hidden_size))
        self.b_o = nn.Parameter(torch.randn(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    # forward pass
    def forward(self, x, states):
        (c_prev, n_prev) = states
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Ensure x is correctly reshaped
        x = x.view(batch_size * seq_len, -1)  # (batch_size * seq_len, input_size)

        # Ensure x is correctly reshaped
        if x.size(1) != self.input_size:
            raise ValueError(f"Expected input size {self.input_size}, but got {x.size(1)}")

        #print(f"x shape: {x.shape}")
        #print(f"w_q shape: {self.w_q.shape}")
        q_t = torch.matmul(x, self.w_q) + self.b_q  # (batch_size * seq_len, hidden_size)
        k_t = (1 / math.sqrt(self.mem_dim)) * (torch.matmul(x, self.w_k) + self.b_k)  # (batch_size * seq_len, mem_dim)
        v_t = torch.matmul(x, self.w_v) + self.b_v  # (batch_size * seq_len, mem_dim)

        i_t = torch.exp(torch.matmul(x, self.w_i) + self.b_i)  # (batch_size * seq_len, 1)
        f_t = torch.sigmoid(torch.matmul(x, self.w_f) + self.b_f)  # (batch_size * seq_len, 1)

        # Ensure v_t and k_t are correctly shaped before torch.ger
        v_t = v_t.view(batch_size * seq_len, self.mem_dim)
        k_t = k_t.view(batch_size * seq_len, self.mem_dim)

        # We need to iterate over each batch to apply torch.ger
        c_t = []
        n_t = []
        for b in range(batch_size * seq_len):
            c_t.append(f_t[b] * c_prev + i_t[b] * torch.ger(v_t[b], k_t[b]))
            n_t.append(f_t[b] * n_prev + i_t[b] * k_t[b].unsqueeze(1))

        c_t = torch.stack(c_t).view(batch_size, seq_len, self.mem_dim, self.mem_dim)
        n_t = torch.stack(n_t).view(batch_size, seq_len, self.mem_dim, 1)

        # Compute h_tilde
        max_nqt = torch.max(torch.abs(torch.matmul(n_t.permute(0, 1, 3, 2), q_t.view(batch_size, seq_len, -1, 1))), torch.tensor(1.0))  # (batch_size, seq_len, hidden_size, 1)
        h_tilde = torch.matmul(c_t, q_t.view(batch_size, seq_len, -1, 1)) / max_nqt  # (batch_size, seq_len, mem_dim, 1)
        h_tilde = h_tilde.squeeze(3)  # (batch_size, seq_len, mem_dim)

        o_t = torch.sigmoid(torch.matmul(x, self.w_o) + self.b_o)  # (batch_size * seq_len, hidden_size)
        #o_t = torch.sigmoid(torch.matmul(x, self.w_o.T) + self.b_o)  # (batch_size * seq_len, hidden_size)
        o_t = o_t.view(batch_size, seq_len, -1)  # (batch_size, seq_len, hidden_size)
        h_t = o_t * h_tilde  # (batch_size, seq_len, hidden_size)
        h_t = h_t.view(batch_size, seq_len, self.hidden_size)
        return h_t, (c_t, n_t)



    def init_hidden(self):
        return (torch.zeros(self.mem_dim, self.mem_dim),
                torch.zeros(self.mem_dim, 1))


# Define the sLSTM module
class sLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(sLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.w_i = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_f = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_o = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_c = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.u_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.u_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.u_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.u_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x, states):
        h_t, c_t, m_t, n_t = states
        i_t = torch.exp(x @ self.w_i + h_t.T @ self.u_i + self.b_i)
        f_t = torch.exp(x @ self.w_f + h_t.T @ self.u_f + self.b_f)
        m_t = torch.max(torch.log(f_t) + m_t, torch.log(i_t))
        i_t_prime = torch.exp(torch.log(i_t) - m_t)
        f_t_prime = torch.exp(torch.log(f_t) + m_t - m_t)
        o_t = torch.sigmoid(x @ self.w_o + h_t.T @ self.u_o + self.b_o)
        z_t = torch.tanh(x @ self.w_c + h_t.T @ self.u_c + self.b_c)
        c_t = f_t_prime * c_t + i_t_prime * z_t
        n_t = f_t_prime * n_t + i_t_prime
        h_t = o_t * (c_t / n_t)
        return h_t, (h_t, c_t, m_t, n_t)

    def init_hidden(self):
        return (torch.zeros(self.hidden_size, 1),
                torch.zeros(self.hidden_size, 1),
                torch.zeros(self.hidden_size, 1),
                torch.zeros(self.hidden_size, 1))


# Define the xLSTM block
class xLSTM_Block(nn.Module):
    def __init__(self, block_type, input_size, hidden_size, layers=2, mem_size=None):
        super(xLSTM_Block, self).__init__()
        # initialize the dropout layer
        self.dropout = DynamicDropout()
        # Create multiple mLSTM and sLSTM layers depending on the block type
        if block_type == 'mLSTM':
            # Create multiple mLSTM layers
            self.layers = nn.ModuleList([mLSTM(input_size if i == 0 else hidden_size, hidden_size, mem_size)
                                         for i in range(layers)])
        elif block_type == 'sLSTM':
            # Create multiple sLSTM layers
            self.layers = nn.ModuleList([sLSTM(hidden_size, hidden_size) for _ in range(layers)])

    # forward pass
    def forward(self, x, initial_states):
        # Initial hidden states
        hidden_states = self.layers[0].init_hidden()
        # Loop through the layers
        for i in range(len(self.layers)):
            # Forward pass through each layer
            x, hidden_states = self.layers[i](x, hidden_states)
            # Apply dropout
            x = self.dropout(x)
        # Return the hidden state and the new states
        return x, hidden_states


# Define the xLSTM model
class xLSTM_model(nn.Module):
    def __init__(self, input_size, hidden_size, mem_size, output_size=1, layers=2):
        super(xLSTM_model, self).__init__()
        # input_size is the number of features in the input
        self.input_size = input_size
        # number of layers in block
        self.layers = layers
        # Initialize xLSTM blocks
        self.blocks = nn.ModuleList()
        # Create xLSTM block for mLSTM
        self.blocks.append(xLSTM_Block('mLSTM', input_size, hidden_size, layers, mem_size))
        # Create xLSTM block for sLSTM
        self.blocks.append(xLSTM_Block('sLSTM', hidden_size, hidden_size, layers))
        # fully connected layer to output the prediction to single value
        self.fc = nn.Linear(hidden_size, output_size)
        # dropout layer
        self.dropout = DynamicDropout()
        # get all the parameters to optimize from all the layers in each block
        self.optimizing_parameters = []
        for i in range(len(self.blocks)):
            for n in range(len(self.blocks[i].layers)):
                self.optimizing_parameters += list(self.blocks[i].layers[n].parameters())

        # optimizer Adam with weight decay for regularization to prevent overfitting
        self.optimizer = torch.optim.Adam(self.optimizing_parameters, lr=0.02, weight_decay=1e-6)

        # self.criterion = nn.MSELoss()
        # self.criterion = nn.L1Loss()
        # L1 loss is more robust to outliers and seams to work best for this problem
        self.criterion = nn.MSELoss()

    # forward pass
    def forward(self, x):
        hidden_states = self.blocks[0].layers[0].init_hidden()
        for i in range(len(self.blocks)):
            x, hidden_states = self.blocks[i](x, hidden_states)
        x = self.fc(x)
        return x
    # Train the model
    def train_model(self, data, val_data, epochs, seq_len):
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=50, verbose=True)

        # Initialize lists to store loss values
        train_losses = []
        val_losses = []
        #test_losses = []

        # loop through the epochs
        for epoch in range(epochs):
            start_time = time.time()  # Start timing
            # zero the gradients
            self.optimizer.zero_grad()
            # initialize the loss
            loss = 0
            # loop through the sequence length
            for t in range(seq_len - 1):
                # get the input at time t
                x = data[:, t]
                # get the target at time t+1
                y_true = data[:, t + 1, 0]
                # get the prediction
                y_pred = self(x)
                # calculate the loss from the training data
                loss += self.criterion(y_pred, y_true)

            # validate the model on the validation data
            val_loss = self.validate(val_data)
            # print the validation loss
            print(f'Epoch {epoch} Validation Loss {val_loss.item()}')
            # record the validation loss
            val_losses.append(val_loss.item())

            # Test the model on the test data
            #test_loss = self.validate(test_data)
            # record the test loss
            #test_losses.append(test_loss.item())

            # call the early stopping object
            early_stopping(val_loss, self)
            # if early stopping is triggered
            if early_stopping.early_stop:
                # print message
                print("Early stopping")
                # stop the training
                break
            # calculate the average loss
            loss.backward()
            # clip the gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.optimizing_parameters, max_norm=1)
            # update the weights
            self.optimizer.step()

            # record the training loss
            train_losses.append(loss.item())

            # print the training loss and elapsed time every 10 epochs
            if epoch % 10 == 0:
                elapsed_time = time.time() - start_time
                print(f'Epoch {epoch} Loss {loss.item()} Elapsed Time: {elapsed_time:.4f} seconds')

        # load the best model before early stopping
        self.load_state_dict(torch.load('checkpoint.pt'))

        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        combined_epochs = list(range(epochs)) + list(range(epochs, 2 * epochs))
        combined_losses = train_losses + val_losses
        plt.plot(combined_epochs, combined_losses, label='Loss')

        # 添加垂直线标注训练集、验证集、测试集的分段
        plt.axvline(x=len(train_losses), color='grey', linestyle='--', label='Train/Val Boundary')
        #plt.axvline(x=len(train_losses) + len(val_losses), color='grey', linestyle='--', label='Val/Test Boundary')

        # 添加标签和标题
        plt.xticks([0, epochs], ['训练集', '验证集'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training-Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Export the loss values to CSV files
        with open('xLSTM_losses.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Train Loss', 'Val Loss'])
            for epoch in range(len(train_losses)):
                writer.writerow([epoch, train_losses[epoch], val_losses[epoch]])

    # predict the model
    def predict(self, data):
        # set the model to evaluation mode
        self.eval()
        # initialize the predictions
        predictions = []
        # loop through the sequence length
        for t in range(data.shape[1] - 1):
            # get the input at time t
            x = data[:, t]
            # get the prediction
            y_pred = self(x)
            # append the prediction to the list
            predictions.append(y_pred.detach().numpy().ravel()[0])
        # set the model back to training mode
        self.train()
        # Export the test data and predictions to a CSV file

        # return the predictions
        return predictions
    # validate the model with the validation data
    def validate(self, data):
        # set the model to evaluation mode
        self.eval()
        # initialize the loss
        loss = 0
        # loop through the sequence length
        for t in range(data.shape[1] - 1):
            # get the input at time t
            x = data[:, t]
            # get the target at time t+1
            y_true = data[:, t + 1, 0]
            # get the prediction
            y_pred = self(x)
            # calculate the loss from the validation data
            loss += self.criterion(y_pred, y_true)
        # set the model back to training mode
        self.train()
        # return the validation loss
        return loss
