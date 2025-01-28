# -*- coding: utf-8 -*-

"""
!nvidia-smi

!gdown --id 16MIleqoIr1vYxlGk4GKnGmrsCPuWkkpT

!unzip -qq ECG5000.zip
"""

import torch
import numpy as np
import time
import sys
import random

# Import my Files
import LSTM_model
import FF_model
import post_training
import pre_training
import training
import plot
import smd_reader
import early_stopping

print(torch.cuda.is_available())
print(torch.version.cuda)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

RANDOM_SEED = random.randint(0, 2 ** 32 - 1)
set_seed = int(sys.argv[4])
if set_seed != 0:
    print(f" Set Seed selected: {set_seed}")
    RANDOM_SEED = int(sys.argv[4])

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

CLASS_NORMAL = 1


def main():
    config = pre_training.load_config()

    start_time = time.time()

    # Use loaded yaml config for data
    train_file = config["data"]["train_file"]
    test_file = config["data"]["test_file"]
    train_split_ratio = config["data"]["train_split_ratio"]
    val_split_ratio = config["data"]["val_split_ratio"]

    # Use loaded yaml config for training variables
    embedding_size = config["training"]["embedding_size"]
    model_path = config["training"]["model_path"]

    # Use command line arguments for choosing model and epochs
    model_used = sys.argv[1]
    epochs = int(sys.argv[2])

    # Initialize variables for post training (threshold and the csv file name for results)
    threshold = config["prediction"]["threshold"]
    results_file_name = 'results/' + sys.argv[3] + '.csv'

    # Decide what dataset to use (ECG or SMD)
    if config["data"]["dataset"] == "ECG":
        df = pre_training.load_files(train_file, test_file)  # load both files into panda dataframe
    elif config["data"]["dataset"] == "SMD":
        df = smd_reader.main(config, RANDOM_SEED)
    else:
        print("Dataset not supported")
        exit(-1)

    # Create train, validation, test normal and test anomaly datasets (also initialize seq_len and n_features
    train_dataset, val_dataset, test_normal_dataset, test_anomaly_dataset, seq_len, n_features = pre_training.data_preprocessing(
        df, train_split_ratio, val_split_ratio, CLASS_NORMAL, RANDOM_SEED)

    print(
        f"Seq_len:  {seq_len}  n_features:  {n_features} Model: {model_used} Epochs: {epochs} Resultfile: {results_file_name}")

    # Select Model to train
    if model_used == "RAE":
        model = LSTM_model.RecurrentAutoencoder(seq_len, n_features, embedding_dim=embedding_size).to(device)
    elif model_used == "FF":
        model = FF_model.FF(1, device=device).to(device)
    else:
        print("no model selected")
        exit(-1)

    # Initialize the early stopping criterion
    early_stopping_crit = early_stopping.EarlyStoppingCriterion(epochs=3, min_diff=0.01)

    # Let the model train
    model, history, last_train_loss, last_val_loss = training.train(model, train_dataset, val_dataset, epochs=epochs,
                                                                    early_stopping_criterion=early_stopping_crit)

    # Plots train and validation loss over epochs
    plot.plot_train_and_val_loss(history)

    # Display the time elapsed during training
    end_time = time.time()
    total_time = round(end_time - start_time, 2)
    print(f'Total time to finish training: {total_time}')

    # Saving the Model
    torch.save(model, model_path)
    _, losses, seq_true_train_list, seq_pred_train_list = post_training.predict(model, train_dataset)

    # Plot a number of random chosen timeseries of traindata
    plot.plot_rdm_true_and_predicted(seq_true_train_list, seq_pred_train_list, "train", 5, RANDOM_SEED, "plots/train")

    # Predict normal test set
    predictions, pred_losses, seq_true_test_list, seq_pred_test_list = post_training.predict(model, test_normal_dataset)

    # Plot a number of random chosen timeseries of normal testdata
    plot.plot_rdm_true_and_predicted(seq_true_test_list, seq_pred_test_list, "test", 5, RANDOM_SEED, "plots/test")

    # Display predictions of normal testdata
    correct_normal = sum(l <= threshold for l in pred_losses)
    print(f'Correct normal predictions: {correct_normal}/{len(test_normal_dataset)}')

    # Create anomaly dataset (shorten length to the test_normal length)
    anomaly_dataset = test_anomaly_dataset[:len(test_normal_dataset)]

    # Predict anomaly test set
    predictions, pred_losses, seq_true_anomaly_list, seq_pred_anomaly_list = post_training.predict(model,
                                                                                                   anomaly_dataset)

    # Plot a number of random chosen timeseries of anomaly testdata
    plot.plot_rdm_true_and_predicted(seq_true_anomaly_list, seq_pred_anomaly_list, "anomaly", 5, RANDOM_SEED,
                                     "plots/anomaly")

    # Display predictions of anomaly testdata
    correct_anomaly = sum(l > threshold for l in pred_losses)
    print(f'Correct anomaly predictions: {correct_anomaly}/{len(anomaly_dataset)}')

    # Initialize data dictionary used of calculating the auc roc score
    data = {
        "test_data": test_normal_dataset + anomaly_dataset,
        "test_labels": np.array([0] * len(test_normal_dataset) + [1] * (len(anomaly_dataset)))
    }

    # Calculate and print the auc roc score
    auc_score = post_training.calculateAUC(model, data)
    print(f'AUC ROC Score: {auc_score}')

    # Write Results in CSV File
    post_training.write_results_to_csv(RANDOM_SEED, model_used, epochs, embedding_size, threshold, last_train_loss,
                                       last_val_loss, correct_normal,
                                       correct_anomaly, total_time, auc_score, results_file_name)


if __name__ == '__main__':
    main()
