import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import get_train_and_val_loaders
from model import get_transformer_model
import numpy as np
import argparse
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import pandas as pd

# Set a fixed random seed for reproducibility in all necessary libraries.
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # Ensures reproducibility for multi-GPU setups.
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train_model(params, plot=False, save_model=False):
    """
    Trains a transformer-based model using given parameters, and optionally plots and saves the model.
    
    Args:
    params (dict): Dictionary containing model hyperparameters.
    plot (bool): If True, plots the training and validation loss curves.
    save_model (bool): If True, saves the model's state_dict.
    
    Returns:
    float: The average validation loss over all epochs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader = get_train_and_val_loaders(
        path_to_hdf5=args.path_to_hdf5,
        path_to_csv=args.path_to_csv,
        hdf5_filename='exams_part17.hdf5',
        batch_size=params['batch_size'],
        val_split=args.val_split,
        subset_size=args.subset_size
    )
    model = get_transformer_model(6, 512, 8, params['nhid'], params['nlayers'], params['dropout']).to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=2)
    scaler = GradScaler()

    train_losses, val_losses = [], []

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
            inputs = inputs.permute(1, 0, 2)
            src_mask = model.generate_square_subsequent_mask(inputs.size(0)).to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs, src_mask)
                loss = criterion(outputs, labels.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad(), autocast():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
                inputs = inputs.permute(1, 0, 2)
                src_mask = model.generate_square_subsequent_mask(inputs.size(0)).to(device)
                outputs = model(inputs, src_mask)
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()

        val_losses.append(val_loss / len(valid_loader))
        scheduler.step(val_losses[-1])

    # Plotting if required
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss_plot.png')
        plt.show()

    # Model saving if required
    if save_model:
        torch.save(model.state_dict(), 'final_model.pth')
        with open("trial_results.txt", "w") as f:
            for key, value in params.items():
                f.write(f"{key}: {value}\n")

    return np.mean(val_losses)

def objective(trial):
    """
    Objective function for hyperparameter tuning using Optuna.
    
    Args:
    trial (optuna.trial): A trial object containing the hyperparameter suggestions.
    
    Returns:
    float: The loss to minimize.
    """
    params = {
        'batch_size': trial.suggest_categorical('batch_size', [16, 24]),
        'lr': trial.suggest_float('lr', 1e-6, 1e-4, log=True),
        'nhid': trial.suggest_categorical('nhid', [512, 1024]),
        'nlayers': trial.suggest_int('nlayers', 2, 4),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5)
    }
    return train_model(params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train neural network')
    parser.add_argument('path_to_hdf5', type=str, help='Path to hdf5 file containing tracings')
    parser.add_argument('path_to_csv', type=str, help='Path to csv file containing annotations')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--epochs', type=int, default=60, help='Number of training epochs')
    parser.add_argument('--subset_size', type=int, default=None, help='Optional size of the subset of data to use for faster iterations')
    args = parser.parse_args()

    study = optuna.create_study(direction='minimize', study_name='ECG Analyser', storage="sqlite:///db.sqlite3", load_if_exists=True)
    study.set_metric_names(["Loss Value"])
    study.optimize(objective, n_trials=0)

    # Print study statistics
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Best trial:")
    trial = study.best_trial
    print(f"    Value: {trial.value}")
    print("    Params: ")
    for key, value in trial.params.items():
        print(f"      {key}: {value}")

    # Retrain and plot with the best parameters
    best_params = trial.params
    train_losses, val_losses= train_model(best_params)