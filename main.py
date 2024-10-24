# -*- coding: utf-8 -*-
"""bachelor_week1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1he_Axo2JbQu2lwzUYV_ABINXC47QODT4
"""

"""
!nvidia-smi

!gdown --id 16MIleqoIr1vYxlGk4GKnGmrsCPuWkkpT

!unzip -qq ECG5000.zip

!pip install -qq pandas

!pip install -qq scipy
torch
"""

from scipy.io import arff

# Commented out IPython magic to ensure Python compatibility.
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn, optim
import time
import sys
import csv
import random

print(torch.cuda.is_available())
print(torch.version.cuda)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RANDOM_SEED = random.randint(0, 2 ** 32 - 1)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

CLASS_NORMAL = 1


def plot_time_series_class(data, class_name, ax, n_steps=10):
    time_series_df = pd.DataFrame(data)

    smooth_path = time_series_df.rolling(n_steps).mean()
    path_deviation = 2 * time_series_df.rolling(n_steps).std()

    under_line = (smooth_path - path_deviation)[0]
    over_line = (smooth_path + path_deviation)[0]

    ax.plot(smooth_path, linewidth=2)
    ax.fill_between(
        path_deviation.index,
        under_line,
        over_line,
        alpha=.125
    )
    ax.set_title(class_name)


def create_dataset(sequences):
    dataset = [torch.tensor(s).unsqueeze(1) for s in sequences]

    n_seq, seq_len, n_features = torch.stack(dataset).shape

    return dataset, seq_len, n_features


"""##Building an LSTM Autoencoder"""


class Encoder(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        return hidden_n.reshape((self.n_features, self.embedding_dim))


class Decoder(nn.Module):

    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )

        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        x = x.repeat(self.seq_len, self.n_features)
        x = x.reshape((self.n_features, self.seq_len, self.input_dim))
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((self.seq_len, self.hidden_dim))
        return self.output_layer(x)


class RecurrentAutoencoder(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class FF(nn.Module):  # Plain Feed Forward Encoder....

    def _getDefaultHPs(self):
        return {"InputSize": 140,
                "LayerSequence": [1, 0.7, 0.3, 0.2, 0.3, 0.7, 1],
                # in multiples of the input ... first and last must be 0
                "ActivationFunction": torch.nn.ReLU,
                "SlicingBeforeFlatten": False,
                # If Slicing is true, the tensors will not be flattend, the tensors will be sliced along the time axis, these slices will than be flattend and concatenated.
                "SliceLength": 0.1,  # of sequence length
                "LayerSequenceLinear": True,
                # if this is true, the layer sequence above will be ignored. instead a linear progression of th layer size to the latent space will be calculated. The layer Sequence will than be overwritten by that.
                "LatentSpaceSize": 0.1,
                "NumLayersPerPart": 3
                }

    def __init__(self, Dimensions, device, **HyperParameters):

        self.device = device
        self.Dimensions = Dimensions

        self.HP = self._getDefaultHPs()
        self.HP.update(HyperParameters)
        nn.Module.__init__(self)

        inputLen = Dimensions * self.HP["InputSize"]
        self.inputLen = inputLen

        if self.HP["SlicingBeforeFlatten"]:
            nWindows = int(np.floor(1 / self.HP["SliceLength"]) + 1)
            windowLength = np.ones(nWindows - 1) * np.floor(self.HP["SliceLength"] * self.inputLen)
            windowLength = np.concatenate([[0], windowLength, [self.inputLen - np.sum(windowLength)]])
            windowEndpoints = np.tril(np.ones(len(windowLength))).dot(windowLength)
            self.windowEndpoints = windowEndpoints.astype(int)

        if self.HP["LayerSequenceLinear"]:
            Layers = np.linspace(1, self.HP["LatentSpaceSize"], self.HP["NumLayersPerPart"])
            Layers = np.concatenate([Layers, np.flip(Layers[:-1])])
            self.HP["LayerSequence"] = Layers

        if len(self.HP["LayerSequence"]) == 1:
            LayerStack = [0] * (3)

            numNeuronsMiddle = int(np.ceil(self.HP["LayerSequence"][0] * inputLen))

            LayerStack[0] = torch.nn.Linear(inputLen, numNeuronsMiddle)
            LayerStack[1] = self.HP["ActivationFunction"]()
            LayerStack[2] = torch.nn.Linear(numNeuronsMiddle, inputLen)

        else:
            LayerStack = [0] * (2 * len(self.HP["LayerSequence"]) - 3)
            # Adding the input/output size to the front and end of the layer stack.
            actualSequence = np.ceil(self.HP["LayerSequence"] * inputLen).astype(int)

            for i in range(0, len(actualSequence) - 2):

                nLastLayer = actualSequence[i]
                if nLastLayer == 0:
                    nLastLayer = 1

                nThisLayer = actualSequence[i + 1]
                if nThisLayer == 0:
                    nThisLayer = 1
                LayerStack[2 * i] = torch.nn.Linear(nLastLayer, nThisLayer)
                LayerStack[2 * i + 1] = self.HP["ActivationFunction"]()
            LayerStack[-1] = torch.nn.Linear(actualSequence[-2], actualSequence[-1])

        # Todo: Clean up and activationfunction as Hyperparameter
        self.model = nn.Sequential(*LayerStack)

        self.model.to(device)

    def forward(self, x):
        xShape = x.shape
        x = x.unsqueeze(0)
        if self.HP["SlicingBeforeFlatten"]:
            snippets = [0] * (len(self.windowEndpoints) - 1)
            snippetShape = [0] * len(snippets)

            for i in range(1, len(self.windowEndpoints)):
                snippets[i - 1] = x[:, :, self.windowEndpoints[i - 1]:self.windowEndpoints[i]]
                snippetShape[i - 1] = snippets[i - 1].shape
                snippets[i - 1] = torch.flatten(snippets[i - 1], start_dim=1)

            x = torch.cat(snippets, -1)

        else:
            x = torch.flatten(x, start_dim=1)
        x = self.model(x)

        if self.HP["SlicingBeforeFlatten"]:
            cutPoints = self.windowEndpoints * self.Dimensions
            reshapedSnippets = [0] * (len(cutPoints) - 1)

            for i in range(1, len(cutPoints)):
                reshapedSnippets[i - 1] = torch.reshape(x[:, cutPoints[i - 1]:cutPoints[i]], snippetShape[i - 1])

            x = torch.cat(reshapedSnippets, -1)
        else:
            x = torch.reshape(x, xShape)
        return x


"""##Training"""


def training(model, train_dataset, val_dataset, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss(reduction='sum').to(device)

    history = dict(train=[], val=[])
    last_train_loss = 0
    last_val_loss = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model = model.train()

        train_losses = []

        for seq_true in train_dataset:
            optimizer.zero_grad()
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)

            loss = criterion(seq_pred, seq_true)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for seq_true in val_dataset:
                seq_true = seq_true.to(device)
                seq_pred = model(seq_true)

                loss = criterion(seq_pred, seq_true)

                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        last_train_loss = train_loss
        val_loss = np.mean(val_losses)
        last_val_loss = val_loss

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        end_time = time.time()
        print(f'Epoch {epoch}: train Loss {train_loss} | val Loss {val_loss} | elapsed time  \
        {round(end_time - start_time, 2)}s')

    return model.eval(), history, last_train_loss, last_val_loss





"""##Chosing a threshold"""


def predict(model, dataset):
    prediction, losses = [], []
    criterion = nn.L1Loss(reduction='sum').to(device)

    with torch.no_grad():
        model = model.eval()
        for seq_true in dataset:
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)

            loss = criterion(seq_pred, seq_true)

            prediction.append(seq_pred.cpu().numpy())
            losses.append(loss.item())

    return prediction, losses


def load_files(train_file, test_file):
    f = arff.loadarff('data/' + train_file)
    train = pd.DataFrame(f[0])

    if test_file != '0':
        f = arff.loadarff('data/' + test_file)
        test = pd.DataFrame(f[0])
        df = pd.concat([train, test])  # concat train- and testfile
    else:
        df = train

    df = df.sample(frac=1.0)  # randomize df

    df['target'] = df['target'].astype(str)  # remove b' ' from target collumn

    return df


def data_preprocessing(df, train_split_ratio, val_split_ratio):
    normal_df = df[df.target == str(CLASS_NORMAL)].drop(labels='target', axis=1)

    # print(normal_df.shape)

    anomaly_df = df[df.target != str(CLASS_NORMAL)].drop(labels='target', axis=1)

    # print(anomaly_df.shape)

    train_df, val_df = train_test_split(normal_df, train_size=train_split_ratio, random_state=RANDOM_SEED)

    val_df, test_df = train_test_split(val_df, train_size=val_split_ratio, random_state=RANDOM_SEED)

    train_sequences = train_df.astype(np.float32).to_numpy().tolist()
    val_sequences = val_df.astype(np.float32).to_numpy().tolist()
    test_sequences = test_df.astype(np.float32).to_numpy().tolist()
    anomaly_sequences = anomaly_df.astype(np.float32).to_numpy().tolist()

    train_dataset, seq_len, n_features = create_dataset(train_sequences)
    val_dataset, seq_len, n_features = create_dataset(val_sequences)
    test_normal_dataset, seq_len, n_features = create_dataset(test_sequences)
    test_anomaly_dataset, seq_len, n_features = create_dataset(anomaly_sequences)

    return train_dataset, val_dataset, test_normal_dataset, test_anomaly_dataset, seq_len, n_features


def calculateAUC(model, data):
    results = np.zeros(len(data["test_data"]))

    lossFunc = torch.nn.L1Loss()

    for i, dataPoint in enumerate(data["test_data"]):
        dataPoint = torch.tensor(dataPoint).to(device)

        result = model(dataPoint)
        loss = lossFunc(dataPoint, result)
        results[i] = loss.item()

    args = np.argsort(results)  # Thanks to user "k.rooijers" on Stackoverflow
    rank = np.argsort(args)

    # Function borrowed form github mlr-org/mlr3measures/R/binary_auc.R
    indices = np.arange(len(data["test_labels"]))[data["test_labels"] != 0]

    n_pos = len(indices)
    n_neg = len(data["test_labels"]) - n_pos

    if n_pos == 0 or n_neg == 0:
        raise Exception("Unable to calculate AUC score. Only elements of one class present.")

    return (np.mean(rank[indices]) - (n_pos + 1) / 2) / n_neg


def write_results_to_csv(model, epochs, embedding_size, threshold, train_loss, val_loss,
                         correct_normal_preds, correct_anomaly_preds, total_time, auc_score,
                         filename="results.csv"):
    header = ['Model', 'Epochs', 'Embedding Size', 'Threshold', 'Random Seed', 'Train Loss', 'Val Loss',
              'Correct Normal Predictions', 'Correct Anomaly Predictions', 'Total Time (s)',
              'AUC Score']

    # Define the row of data that will be written to the CSV
    row = [model, epochs, embedding_size, threshold, RANDOM_SEED, train_loss, val_loss, correct_normal_preds,
           correct_anomaly_preds, total_time, auc_score]

    # Check if the file already exists, and if it does not, write the header first
    try:
        with open(filename, 'x', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)  # Write header first
    except FileExistsError:
        pass  # File already exists, so we don't need to write the header again

    # Now write the actual data to the CSV file
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)

    print(f'Results written to {filename} successfully!')


# Main

def main():
    start_time = time.time()

    # config
    train_file = 'ECG5000_TRAIN.arff'
    test_file = 'ECG5000_TEST.arff'
    train_split_ratio = 0.85
    val_split_ratio = 0.5
    embedding_size = 64
    model_path = 'model.pth'
    model_used = sys.argv[1]
    epochs = int(sys.argv[2])
    threshold = 26
    results_file_name = sys.argv[3] + '.csv'

    df = load_files(train_file, test_file)  # load both files into panda dataframe

    train_dataset, val_dataset, test_normal_dataset, test_anomaly_dataset, seq_len, n_features = data_preprocessing \
        (df, train_split_ratio, val_split_ratio)
    print(f"Seq_len:  {seq_len}  n_features:  {n_features}")
    if model_used == "RAE":
        model = RecurrentAutoencoder(seq_len, n_features, embedding_dim=embedding_size).to(device)
    elif model_used == "FF":
        model = FF(1, device=device).to(device)
    else:
        print("no model selected")
        exit(-1)
    with torch.profiler.profile(with_stack=True) as prof:
        model, history, last_train_loss, last_val_loss = training(model, train_dataset, val_dataset, epochs=epochs)

    end_time = time.time()
    total_time = round(end_time - start_time, 2)
    print(f'Total time to finish training: {total_time}')

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Saving the Model
    torch.save(model, model_path)
    _, losses = predict(model, train_dataset)

    # Evaluation
    predictions, pred_losses = predict(model, test_normal_dataset)

    correct_normal = sum(l <= threshold for l in pred_losses)
    print(f'Correct normal predictions: {correct_normal}/{len(test_normal_dataset)}')

    anomaly_dataset = test_anomaly_dataset[:len(test_normal_dataset)]
    predictions, pred_losses = predict(model, anomaly_dataset)
    correct_anomaly = sum(l > threshold for l in pred_losses)
    print(f'Correct anomaly predictions: {correct_anomaly}/{len(anomaly_dataset)}')

    data = {
        "test_data": test_normal_dataset + anomaly_dataset,
        "test_labels": np.array([0] * len(test_normal_dataset) + [1] * (len(anomaly_dataset)))
    }

    print(type(data))
    auc_score = calculateAUC(model, data)  # Calculate AUC Score
    print(f'AUC ROC Score: {auc_score}')

    # Write Results in CSV File
    write_results_to_csv(model_used, epochs, embedding_size, threshold, last_train_loss, last_val_loss, correct_normal,
                         correct_anomaly, total_time, auc_score, results_file_name)


if __name__ == '__main__':
    main()
