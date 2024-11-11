import torch
import numpy as np
import pandas as pd
import yaml
from scipy.io import arff
from sklearn.model_selection import train_test_split


def load_config(path="config.yaml"):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


def load_files(train_file, test_file):
    f = arff.loadarff('data/' + train_file)
    train = pd.DataFrame(f[0])

    if test_file != '0':
        f = arff.loadarff('data/' + test_file)
        test = pd.DataFrame(f[0])
        df = pd.concat([train, test])  # concat train- and testfile
    else:
        df = train

    # df = df.sample(frac=1.0)  # randomize df
    df['target'] = df['target'].apply(lambda x: x.decode('utf-8'))  # remove b' ' from target collumn
    return df


def create_dataset(sequences):
    dataset = [torch.tensor(s).unsqueeze(1) for s in sequences]

    n_seq, seq_len, n_features = torch.stack(dataset).shape

    return dataset, seq_len, n_features


def data_preprocessing(df, train_split_ratio, val_split_ratio, CLASS_NORMAL, RANDOM_SEED):
    normal_df = df[df.target == str(CLASS_NORMAL)].drop(labels='target', axis=1)
    anomaly_df = df[df.target != str(CLASS_NORMAL)].drop(labels='target', axis=1)

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
