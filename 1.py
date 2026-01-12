import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from model_5 import RNNModel, LSTMModel, GRUModel, CNNModel, MLPModel
import matplotlib.pyplot as plt

# Define a function to extract features from the data
def extract_features(data, features, seq_len):
    # Get the specified features
    data = data[features]
    data = data.interpolate(method='linear')

    # Find the minimum and maximum values of the data
    data_min = data.min()
    data_max = data.max()

    # Normalize the data
    data = (data - data_min) / (data_max - data_min)

    # Create sequences of the specified length
    sequences = []
    for i in range(len(data) - seq_len + 1):
        sequences.append(data.iloc[i:i + seq_len].values)

    sequences = np.array(sequences)

    # Split data into train (60%), val (20%), and test (20%) sets
    train, val, test = (sequences[:int(0.6 * len(sequences))],
                        sequences[int(0.6 * len(sequences)):int(0.8 * len(sequences))],
                        sequences[int(0.8 * len(sequences)):])

    # Convert to PyTorch tensors
    train_tensor = torch.tensor(train).float()
    val_tensor = torch.tensor(val).float()
    test_tensor = torch.tensor(test).float()

    # Return the tensors and the min/max values
    return train_tensor, val_tensor, test_tensor, data_min, data_max

# Define a function to denormalize the predictions
def denormalize(predictions, data_min, data_max):
    # Convert the predictions to a numpy array
    predictions = np.array(predictions)
    # Denormalize the predictions using the min/max values
    return predictions * (data_max[0] - data_min[0]) + data_min[0]

# Define the main function
def main():
    data = pd.read_csv('mini_AU2408test_10000.csv')

    print(data.head())
    data = data.interpolate(method='linear')
    print(data.isna().sum())  # Print the number of NaN values in each column

    # Create evaluation data for the models
    x = data['last'].values
    test = x[int(0.8 * len(data)):]

    seq_len = 80

    # Univariate data preparation
    uni_features = ['last']
    uni_train, uni_val, uni_test, uni_min, uni_max = extract_features(data, uni_features, seq_len)

    # Multivariate data preparation
    multi_features = ['last', 'high', 'low']
    multi_train, multi_val, multi_test, multi_min, multi_max = extract_features(data, multi_features, seq_len)

    # Validate the shapes of the data
    print(f"Univariate Test Shape: {uni_test.shape}")
    print(f"Multivariate Test Shape: {multi_test.shape}")

    error_test = np.array(test[-len(uni_test):])

    hidden_size = 30
    layers = 1
    input_size = len(uni_features)
    output_size = 1

    models = {
        'RNN': RNNModel(input_size, hidden_size, output_size),
        'LSTM': LSTMModel(input_size, hidden_size, output_size),
        'GRU': GRUModel(input_size, hidden_size, output_size),
        'CNN': CNNModel(uni_train.shape[2], [32, 16], output_size, seq_len),  # Updated input_dim to seq_len
        'MLP': MLPModel(input_size * seq_len, [32, 16], output_size)
    }

    predictions = {}

    for name, model in models.items():
        print(f"Training {name} model...")
        # Assuming each model has a similar training process
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        epochs = 50  # Adjust the number of epochs as needed

        # Train the model
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(uni_train)
            loss = criterion(output, uni_train[:, -1, :output.shape[1]])  # Match output shape
            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_output = model(uni_val)
                val_loss = criterion(val_output, uni_val[:, -1, :val_output.shape[1]])

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, {name} model, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}")

        # Predict
        model.eval()
        with torch.no_grad():
            predictions[name] = model(uni_test).numpy().flatten()

    # Denormalize predictions
    for name in predictions:
        predictions[name] = denormalize(predictions[name], uni_min, uni_max)

    # Calculate metrics and plot
    plt.title('Model Comparison')
    plt.plot(error_test, color='purple', label='Test Data')

    for name, pred in predictions.items():
        plt.plot(pred, label=f'{name} Model')

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
