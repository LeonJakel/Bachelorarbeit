import torch
from torch import nn
import time
import numpy as np

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


def train(model, train_dataset, val_dataset, epochs, early_stopping_criterion=None):
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

        # Check Early Stopping
        if early_stopping_criterion and early_stopping_criterion.should_stop(history):
            print('Early stopping')
            break

    return model.eval(), history, last_train_loss, last_val_loss
