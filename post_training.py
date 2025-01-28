import torch
from torch import nn
import numpy as np
import csv
import plot

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


def predict(model, dataset):
    prediction, losses, = [], []
    seq_true_list, seq_pred_list = [], []
    criterion = nn.L1Loss(reduction='sum').to(device)

    with torch.no_grad():
        model = model.eval()



        for seq_true in dataset:


            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)


            seq_true_list.append(seq_true)
            seq_pred_list.append(seq_pred)

            loss = criterion(seq_pred, seq_true)

            prediction.append(seq_pred.cpu().numpy())
            losses.append(loss.item())

    return prediction, losses, seq_true_list, seq_pred_list


def calculateAUC(model, data):
    results = np.zeros(len(data["test_data"]))

    lossFunc = torch.nn.L1Loss()

    for i, dataPoint in enumerate(data["test_data"]):
        dataPoint = dataPoint.clone().detach().to(device)

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


def write_results_to_csv(RANDOM_SEED, model, epochs, embedding_size, threshold, train_loss, val_loss,
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
