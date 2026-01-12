# Import Meteostat library and dependencies
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from math import sqrt
from math import pi
import matplotlib.pyplot as plt
from meteostat import Point, Daily
import numpy as np
import torch
import xLSTM
import PCA_method
from model_5 import RNNModel, LSTMModel, GRUModel, CNNModel, MLPModel
import pandas as pd
import markov as mk
import csv

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文

def create_time_windows(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)
window_size = 10

# Define a function to extract features from the data
def extract_features(data, features, variance_threshold=0.90, window_size=80):
    try:
        # Get the specified features
        data = data[features]

        # Find the minimum and maximum values of the data
        data_min = data.min()
        data_max = data.max()

        # Normalize the data
        data = (data - data_min) / (data_max - data_min)

        print(f"Original data shape: {data.shape}")

        # Apply PCA
        pca_data, data_mean, data_std, explained_variance, n_components = PCA_method.apply_pca(data, variance_threshold)

        print(f"PCA data shape: {pca_data.shape}, n_components: {n_components}")
        print(f"Explained variance by components: {explained_variance}")

        if n_components > 1:
            # Calculate VIF for the PCA-transformed data
            vif_data = PCA_method.calculate_vif(pca_data)
            print("VIF after PCA:")
            print(vif_data)
        else:
            print("Single component, VIF calculation skipped.")

        # Create time windows
        X, y = create_time_windows(pca_data, window_size)
        print(f"Time windows X shape: {X.shape}, y shape: {y.shape}")

        # Check if there is enough data to create at least one window
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Not enough data to create time windows.")

        # Split the data into train, val, and test sets
        split1 = int(0.6 * len(X))
        split2 = int(0.8 * len(X))
        X_train, X_val, X_test = X[:split1], X[split1:split2], X[split2:]
        y_train, y_val, y_test = y[:split1], y[split1:split2], y[split2:]

        print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}, X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}, y_val shape: {y_val.shape}, y_test shape: {y_test.shape}")

        # Convert to PyTorch tensors
        train_tensor = torch.tensor(X_train).float()
        val_tensor = torch.tensor(X_val).float()
        test_tensor = torch.tensor(X_test).float()

        return train_tensor, val_tensor, test_tensor, data_min, data_max
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None, None

'''train, val, test = (pca_data[:int(0.6 * len(pca_data))],
                        pca_data[int(0.6 * len(pca_data)):int(0.8 * len(pca_data))],
                        pca_data[int(0.8 * len(pca_data)):])

    seq_len = 80
    # Ensure the length of data is a multiple of seq_len
    train_len = (len(train) // seq_len) * seq_len
    val_len = (len(val) // seq_len) * seq_len
    test_len = (len(test) // seq_len) * seq_len

    train = train[:train_len]
    val = val[:val_len]
    test = test[:test_len]

    batch_size_train = len(train) // seq_len
    batch_size_val = len(val) // seq_len
    batch_size_test = len(test) // seq_len

    print(f"train length: {len(train)}, batch_size_train: {batch_size_train}")
    print(f"val length: {len(val)}, batch_size_val: {batch_size_val}")
    print(f"test length: {len(test)}, batch_size_test: {batch_size_test}")

    # Reshape data to (batch_size, seq_len, n_components)
    train_tensor = torch.tensor(train).float().view(batch_size_train, seq_len, n_components)
    val_tensor = torch.tensor(val).float().view(batch_size_val, seq_len, n_components)
    test_tensor = torch.tensor(test).float().view(batch_size_test, seq_len, n_components)'''

'''train_tensor = torch.tensor(train).float()
    val_tensor = torch.tensor(val).float()
    test_tensor = torch.tensor(test).float()'''

# Define a function to extract other 5 models' features from the data
def extract_5_features(data, features, seq_len):
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

# Define a function to denormalize the predictions 反归一化
def denormalize(predictions, data_min, data_max):
    # Convert the predictions to a numpy array
    predictions = np.array(predictions)
    # Denormalize the predictions using the min/max values
    return predictions * (data_max[0] - data_min[0]) + data_min[0]


# Define a function to create xLSTM models
def create_model(train, val, test, hidden, mem, layers, seq):
    # Define the input size, hidden size, memory dimension, and sequence length
    input_size = train.shape[2]
    hidden_size = hidden
    mem_dim = mem
    seq_len = seq

    # Create the xLSTM model
    model = xLSTM.xLSTM_model(input_size, hidden_size, mem_dim, 1, layers)
    # Train the model on the training set and validate on the validation set
    # for 200 epochs with the specified sequence length
    model.train_model(train, val, 3, seq_len)

    # Training might have been stopped early if the validation loss did not improve
    # Load the best model from the checkpoint
    model.load_state_dict(torch.load('checkpoint.pt'))

    # Make predictions on the test set
    test_output = model.predict(test)

    # Return the predictions
    return test_output


# Define the main function
def main():
    data = pd.read_csv('mini_AU2408test_10000.csv')
    print(data.head())

    data = data.interpolate(method='linear')
    # Create evaluation data for the models
    x = data['last'].values
    # split a univariate sequence into samples, train (60%), val (20%), test (20%)
    test = x[int(0.8 * len(data)):]

    # univariate data preparation
    uni_features = ['last']
    # Extract the features from the data and normalize them
    uni_train, uni_val, uni_test, uni_min, uni_max = extract_features(data, uni_features)

    # multivariate data preparation
    multi_features = ['last','high', 'low']
    # Extract the features from the data and normalize them
    multi_train, multi_val, multi_test, multi_min, multi_max = extract_features(data, multi_features)
    print(uni_test.isnan().sum())
    print(uni_train.isnan().sum())
    print(multi_test.isnan().sum())
    print(multi_train.isnan().sum())
    # Evaluate the models on the test set
    error_test = np.array(test)
    # Cut the error_test array to match the length of the predictions
    error_test = error_test[:len(uni_test[0])]

    # xLSTM model parameters
    hidden_size = 30
    mem_dim = 30
    layers = 1
    seq_len = 80

    # Create the xLSTM models
    # Univariate model, create_model function returns the predictions
    uni_model_pred = create_model(uni_train, uni_val, uni_test, hidden_size, mem_dim, layers, seq_len)
    # Multivariate model, create_model function returns the predictions
    multi_model_pred = create_model(multi_train, multi_val, multi_test, hidden_size, mem_dim, layers, seq_len)

    # Denormalize the predictions
    denorm_uni_pred = denormalize(uni_model_pred, uni_min, uni_max)
    denorm_multi_pred = denormalize(multi_model_pred, multi_min, multi_max)

    # Split the data into training (80%) and test sets (20%) for the Markov Chain model
    markov_train, markov_test = train_test_split(x, test_size=0.2, shuffle=False)

    # Convert the data to integers for the Markov Chain model
    # This narrows down the state space and makes the model more accurate
    markov_train = markov_train.round().astype(int)
    markov_test = markov_test.round().astype(int)

    # Find the best order for the Markov Chain model between 2 and 30
    best_order = mk.find_best_order(markov_train, markov_test, 2, 30)

    # Create and train the Markov Chain model with the best order
    model = mk.MarkovChain(best_order)
    model.fit(markov_train)

    # Add the last 'order' elements of the training data to the test data
    # to ensure that the test data is long enough for the predictions
    # to start of the test data in the LSTM models
    markov_test = np.concatenate((markov_train[-best_order:], markov_test))

    # Make predictions on the test set, feeding the model order elements at a time
    markov_pred = [model.predict(markov_test[i - best_order:i]) for i in range(best_order, len(markov_test))]

    # Univariate data preparation
    uni_5_features = ['last']
    uni_5_train, uni_5_val, uni_5_test, uni_5_min, uni_5_max = extract_5_features(data, uni_5_features, seq_len)

    # Multivariate data preparation
    # multi_features = ['last', 'high', 'low']
    # multi_train, multi_val, multi_test, multi_min, multi_max = extract_features(data, multi_features, seq_len)

    # Validate the shapes of the data
    # print(f"Univariate Test Shape: {uni_test.shape}")
    # print(f"Multivariate Test Shape: {multi_test.shape}")

    error_5_test = np.array(test[-len(uni_5_test):])
    input_size = len(uni_5_features)
    output_size = 1

    models = {
        'RNN': RNNModel(input_size, hidden_size, output_size),
        'LSTM': LSTMModel(input_size, hidden_size, output_size),
        'GRU': GRUModel(input_size, hidden_size, output_size),
        'CNN': CNNModel(uni_train.shape[2], [32, 16], output_size, seq_len),  # Updated input_dim to seq_len
        'MLP': MLPModel(input_size * seq_len, [32, 16], output_size)
    }

    predictions = {}

    # Initialize dictionaries to store loss values
    train_losses = {name: [] for name in models.keys()}
    val_losses = {name: [] for name in models.keys()}
    #test_losses = {name : [] for name in models.keys()}

    for name, model in models.items():
        print(f"Training {name} model...")
        # Assuming each model has a similar training process
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        epochs = 10  # Adjust the number of epochs as needed

        # Train the model
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(uni_5_train)
            loss = criterion(output, uni_5_train[:, -1, :output.shape[1]])  # Match output shape
            loss.backward()
            optimizer.step()

            # Record training loss
            train_losses[name].append(loss.item())

            # Validation
            model.eval()
            with torch.no_grad():
                val_output = model(uni_5_val)
                val_loss = criterion(val_output, uni_5_val[:, -1, :val_output.shape[1]])

            # Record validation loss
            val_losses[name].append(val_loss.item())

            # Test
            #with torch.no_grad():
            #    test_output = model(uni_5_test)
            #    test_loss = criterion(test_output, uni_5_test[:, -1, :test_output.shape[1]])

            # Record test loss
           # test_losses[name].append(test_loss.item())

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, {name} model, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}")

        # Predict
        model.eval()
        with torch.no_grad():
            predictions[name] = model(uni_5_test).numpy().flatten()

    # Export the loss values to CSV files
    for name in models.keys():
        with open(f'{name}_losses.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Train Loss', 'Val Loss'])
            for epoch in range(epochs):
                writer.writerow([epoch, train_losses[name][epoch], val_losses[name][epoch]])
    # 绘制所有模型的损失曲线
    plt.figure(figsize=(12, 8))
    for name in models.keys():
        combined_epochs = list(range(epochs)) + list(range(epochs, 2 * epochs))
        combined_losses = train_losses[name] + val_losses[name]
        plt.plot(combined_epochs, combined_losses, label=f'{name} Loss')

    # 添加垂直线标注训练集、验证集、测试集的分段
    plt.axvline(x=epochs, color='grey', linestyle='--', label='Train/Val Boundary')
    #plt.axvline(x=epochs * 2, color='grey', linestyle='--', label='Val/Test Boundary')

    # 添加标签和标题
    plt.xticks([0, epochs], ['训练集', '验证集'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training-Validation Loss for 5 other Models')
    plt.legend()
    plt.grid(True)
    plt.show()


    # Denormalize predictions
    for name in predictions:
        predictions[name] = denormalize(predictions[name], uni_5_min, uni_5_max)

    # Define the Dstat calculation function
    #def calculate_dstat(actual, predicted):
    #    return sum((actual[:-1] - actual[1:]) * (predicted[:-1] - predicted[1:]) > 0) / (len(actual) - 1)

    # Define MSPE calculation function
    def mean_square_percent_error(y_true, y_pred):
        return np.mean(((y_true - y_pred) / y_true) ** 2) * 100

    # Define MAPE calculation function
    def mean_absolute_percent_error(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Define the Dstat calculation function
    def calculate_dstat(actual, predicted):
        actual = np.array(actual)
        predicted = np.array(predicted)
        return sum((actual[:-1] - actual[1:]) * (predicted[:-1] - predicted[1:]) > 0) / (len(actual) - 1)

    # Calculate the metrics for the models
    metrics = {}
    for name, pred in predictions.items():
        metrics[name] = {
            'mae': mean_absolute_error(error_5_test[:len(pred)], pred),
            'rmse': sqrt(mean_squared_error(error_5_test[:len(pred)], pred)),
            'corr': pearsonr(error_5_test[:len(pred)], pred)[0],
            'r2': r2_score(error_5_test[:len(pred)], pred),
            'mspe': mean_square_percent_error(error_5_test[:len(pred)], pred),
            'mape': mean_absolute_percent_error(error_5_test[:len(pred)], pred),
            'dstat': calculate_dstat(error_5_test[:len(pred)], pred)
        }

    # Calculate metrics for additional models
    metrics['xLSTM Univariate'] = {
        'mae': mean_absolute_error(error_test[:len(uni_model_pred)], denorm_uni_pred),
        'rmse': sqrt(mean_squared_error(error_test[:len(uni_model_pred)], denorm_uni_pred)),
        'corr': pearsonr(error_test[:len(uni_model_pred)], denorm_uni_pred)[0],
        'r2': r2_score(error_test[:len(uni_model_pred)], denorm_uni_pred),
        'dstat': calculate_dstat(error_test[:len(uni_model_pred)], denorm_uni_pred),
        'mspe': mean_square_percent_error(error_test[:len(uni_model_pred)], denorm_uni_pred),
        'mape': mean_absolute_percent_error(error_test[:len(uni_model_pred)], denorm_uni_pred)
    }

    metrics['xLSTM Multivariate'] = {
        'mae': mean_absolute_error(error_test[:len(uni_model_pred)], denorm_multi_pred),
        'rmse': sqrt(mean_squared_error(error_test[:len(uni_model_pred)], denorm_multi_pred)),
        'corr': pearsonr(error_test[:len(uni_model_pred)], denorm_multi_pred)[0],
        'r2': r2_score(error_test[:len(uni_model_pred)], denorm_multi_pred),
        'dstat': calculate_dstat(error_test[:len(uni_model_pred)], denorm_multi_pred),
        'mspe': mean_square_percent_error(error_test[:len(uni_model_pred)], denorm_multi_pred),
        'mape': mean_absolute_percent_error(error_test[:len(uni_model_pred)], denorm_multi_pred)
    }

    metrics['Markov Chain'] = {
        'mae': mean_absolute_error(error_test[:len(uni_model_pred) - best_order + 1], markov_pred[best_order:]),
        'rmse': sqrt(mean_squared_error(error_test[:len(uni_model_pred) - best_order + 1], markov_pred[best_order:])),
        'corr': pearsonr(error_test[:len(uni_model_pred) - best_order + 1], markov_pred[best_order:])[0],
        'r2': r2_score(error_test[:len(uni_model_pred) - best_order + 1], markov_pred[best_order:]),
        'dstat': calculate_dstat(error_test[:len(uni_model_pred) - best_order + 1], markov_pred[best_order:]),
        'mspe': mean_square_percent_error(error_test[:len(uni_model_pred) - best_order + 1], markov_pred[best_order:]),
        'mape': mean_absolute_percent_error(error_test[:len(uni_model_pred) - best_order + 1], markov_pred[best_order:])
    }

    # Print the metrics for the models
    for name, metric in metrics.items():
        print(f'MAE for {name}: {metric["mae"]:.3f}')
        print(f'RMSE for {name}: {metric["rmse"]:.3f}')
        print(f'Correlation for {name}: {metric["corr"]:.3f}')
        print(f'R-Square for {name}: {metric["r2"]:.3f}')
        print(f'Dstat for {name}: {metric["dstat"]:.3f}')
        print(f'MSPE for {name}: {metric["mspe"]:.3f}')
        print(f'MAPE for {name}: {metric["mape"]:.3f}')
        print('')

    # Normalize the data for the radar plot
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min())

    df1 = pd.DataFrame(metrics).T
    df_norm = df1.apply(normalize)

    # Create the radar chart
    categories = list(df_norm.columns)
    num_categories = len(categories)

    angles = [n / float(num_categories) * 2 * pi for n in range(num_categories)]
    angles += angles[:1]

    plt.figure(figsize=(10, 10))

    ax = plt.subplot(111, polar=True)

    for model in df_norm.index:
        values = df_norm.loc[model].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=model)
        ax.fill(angles, values, alpha=0.1)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.title('Model Evaluation Metrics Radar Chart')
    plt.show()

    # 确保所有序列的长度一致
    all_sequences = [error_test, denorm_uni_pred, denorm_multi_pred, markov_pred[best_order:]]
    all_sequences.extend([predictions[name] for name in predictions])
    length = min(map(len, all_sequences))

    # 截取相同长度
    error_test = error_test[:length]
    denorm_uni_pred = denorm_uni_pred[:length]
    denorm_multi_pred = denorm_multi_pred[:length]
    markov_pred = markov_pred[best_order:best_order + length]

    for name in predictions:
        predictions[name] = predictions[name][:length]

    #导出到csv文件
    with open('test_predictions_combined.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['Time Step', 'True Value', 'xLSTM Univariate Model', 'xLSTM Multivariate Model', 'Markov Chain Model']
        header.extend(predictions.keys())  # Add other model names to the header
        writer.writerow(header)

        for t in range(len(error_test)):
            row = [t, error_test[t], denorm_uni_pred[t], denorm_multi_pred[t], markov_pred[t]]
            row.extend([predictions[name][t] for name in predictions])
            writer.writerow(row)

    # Plot the predictions
    plt.title('Model Comparison')
    plt.plot(error_test, color='purple',label='Test Data')
    plt.plot(denorm_uni_pred, label='xLSTM Univariate Model')
    plt.plot(denorm_multi_pred, label='xLSTM Multivariate Model')
    plt.plot(markov_pred, label='Markov Chain Model')

    for name, pred in predictions.items():
        plt.plot(pred, label=f'{name} Model')

    plt.legend()
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()