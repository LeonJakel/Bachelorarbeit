import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_time_series(tensor, title="Timeseries", xlabel="Time", ylabel="Value", y_min=-5, y_max=3, y_step=1):
    tensor = tensor.squeeze()

    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input tensor must be of type torch.Tensor.")

    if tensor.dim() != 1:
        raise ValueError("Tensor must be 1 dimensional.")

    data = tensor.cpu().numpy()  # Convert Tensor to numpy

    # Creating the Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data, color="blue", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)

    ax.set_ylim([y_min, y_max])
    ax.set_yticks(range(y_min, y_max + 1, y_step))

    return fig, ax


def save_plot(plot, name, directory="plots"):
    filepath = f"{directory}/{name}.png"
    plt.savefig(filepath)
    plt.close(plot)


def plot_rdm_true_and_predicted(plot_list_true, plot_list_pred, name, number_of_plots, RANDOM_SEED, directory="plots"):
    np.random.seed(RANDOM_SEED)
    rdm_plot_list = np.random.randint(low=0, high=len(plot_list_true) - 1, size=number_of_plots)

    for i, index in enumerate(rdm_plot_list):
        plot_name = f"{name}_true_{i}"
        fig, ax = plot_time_series(plot_list_true[index], title=plot_name, xlabel="Time", ylabel="Value")
        ax.set_ylim(0, 1)
        save_plot(fig, plot_name, directory)

    for i, index in enumerate(rdm_plot_list):
        plot_name = f"{name}_pred_{i}"
        fig, ax = plot_time_series(plot_list_pred[index], title=plot_name, xlabel="Time", ylabel="Value")
        ax.set_ylim(0, 1)
        save_plot(fig, plot_name, directory)


def plot_train_and_val_loss(history):
    hlp_train = history["train"]

    hlp_val = history["val"]

    train_losses = torch.tensor(history['train'])
    val_losses = torch.tensor(history['val'])

    fig, ax = plot_time_series(train_losses, title="Training Loss over Epochs", xlabel="Epoch", ylabel="Train Loss", y_min = int(hlp_train[-1])-5, y_max=int(hlp_train[0])+5, y_step=2)
    save_plot(fig, name="train_loss", directory="plots/other")

    fig, ax = plot_time_series(val_losses, title="Validation Loss over Epochs", xlabel="Epoch", ylabel="Validation Loss",y_min = int(hlp_train[-1])-5, y_max=int(hlp_train[0])+5, y_step=2)
    save_plot(fig, name="val_loss", directory="plots/other")