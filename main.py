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

print(torch.cuda.is_available())
print(torch.version.cuda)

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    # config
    train_file = config["data"]["train_file"]
    test_file = config["data"]["test_file"]
    train_split_ratio = config["data"]["train_split_ratio"]
    val_split_ratio = config["data"]["val_split_ratio"]

    embedding_size = config["training"]["embedding_size"]
    model_path = config["training"]["model_path"]
    model_used = sys.argv[1]
    epochs = int(sys.argv[2])

    threshold = config["prediction"]["threshold"]
    results_file_name = 'results/' + sys.argv[3] + '.csv'

    df = pre_training.load_files(train_file, test_file)  # load both files into panda dataframe

    train_dataset, val_dataset, test_normal_dataset, test_anomaly_dataset, seq_len, n_features = pre_training.data_preprocessing(
        df, train_split_ratio, val_split_ratio, CLASS_NORMAL, RANDOM_SEED)
    print(
        f"Seq_len:  {seq_len}  n_features:  {n_features} Model: {model_used} Epochs: {epochs} Resultfile: {results_file_name}")
    if model_used == "RAE":
        model = LSTM_model.RecurrentAutoencoder(seq_len, n_features, embedding_dim=embedding_size).to(device)
    elif model_used == "FF":
        model = FF_model.FF(1, device=device).to(device)
    else:
        print("no model selected")
        exit(-1)

    model, history, last_train_loss, last_val_loss = training.train(model, train_dataset, val_dataset, epochs=epochs)

    end_time = time.time()
    total_time = round(end_time - start_time, 2)
    print(f'Total time to finish training: {total_time}')

    # Saving the Model
    torch.save(model, model_path)
    _, losses = post_training.predict(model, train_dataset)

    # Evaluation
    predictions, pred_losses = post_training.predict(model, test_normal_dataset)

    correct_normal = sum(l <= threshold for l in pred_losses)
    print(f'Correct normal predictions: {correct_normal}/{len(test_normal_dataset)}')

    anomaly_dataset = test_anomaly_dataset[:len(test_normal_dataset)]
    predictions, pred_losses = post_training.predict(model, anomaly_dataset)
    correct_anomaly = sum(l > threshold for l in pred_losses)
    print(f'Correct anomaly predictions: {correct_anomaly}/{len(anomaly_dataset)}')

    data = {
        "test_data": test_normal_dataset + anomaly_dataset,
        "test_labels": np.array([0] * len(test_normal_dataset) + [1] * (len(anomaly_dataset)))
    }

    auc_score = post_training.calculateAUC(model, data)  # Calculate AUC Score
    print(f'AUC ROC Score: {auc_score}')

    # Write Results in CSV File
    post_training.write_results_to_csv(RANDOM_SEED, model_used, epochs, embedding_size, threshold, last_train_loss,
                                       last_val_loss, correct_normal,
                                       correct_anomaly, total_time, auc_score, results_file_name)


if __name__ == '__main__':
    main()
