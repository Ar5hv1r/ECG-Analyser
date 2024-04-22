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

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # if using multi-GPU.
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train_model(params, plot=False, save_model=False):
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

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
    if save_model:
        torch.save(model.state_dict(), 'final_model.pth')
        # Optionally save hyperparameters
        with open("trial_results.txt", "w") as f:
            for key, value in params.items():
                f.write(f"{key}: {value}\n")

    return np.mean(val_losses)

def objective(trial):
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
    parser.add_argument('path_to_hdf5', type=str, help='path to hdf5 file containing tracings')
    parser.add_argument('path_to_csv', type=str, help='path to csv file containing annotations')
    parser.add_argument('--val_split', type=float, default=0.2, help='validation split ratio')
    parser.add_argument('--epochs', type=int, default=60, help='number of training epochs')
    parser.add_argument('--subset_size', type=int, default=None, help='Optional size of the subset of data to use for faster iterations')
    args = parser.parse_args()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=1)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Best trial:")
    trial = study.best_trial
    print(f"    Value: {trial.value}")
    print("    Params: ")
    for key, value in trial.params.items():
        print(f"      {key}: {value}")

    # Retrain and plot with best parameters
    best_params = trial.params
    train_model(best_params, plot=True, save_model=True)


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
# import numpy as np
# import argparse
# from datasets import get_train_and_val_loaders
# from model import get_transformer_model
# import matplotlib.pyplot as plt
# from torch.cuda.amp import autocast, GradScaler
# from torch.optim.lr_scheduler import StepLR
# from ray import tune

# SEED = 42
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)  # if using multi-GPU.
# np.random.seed(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# scaler = GradScaler()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Train neural network')
#     # define command line args
#     parser.add_argument('path_to_hdf5', type=str,
#                         help='path to hdf5 file containing tracings')
#     parser.add_argument('path_to_csv', type=str,
#                         help='path to csv file containing annotations')
#     parser.add_argument('--val_split', type=float,
#                         default=0.2, help='validation split ratio')
#     parser.add_argument('--batch_size', type=int,
#                         default=24, help='training batch size')
#     parser.add_argument('--epochs', type=int, default=50,
#                         help='number of training epochs')
#     parser.add_argument('--lr', type=float, default=6.303597746005374e-06, help='learning rate')
#     parser.add_argument('--subset_size', type=int, default=None, help='Optional size of the subset of data to use for faster iterations')
#     args = parser.parse_args()

#     # set up device (CPU/GPU)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
#     train_loader, valid_loader = get_train_and_val_loaders(
#     path_to_hdf5=args.path_to_hdf5,
#     path_to_csv=args.path_to_csv,
#     hdf5_filename='exams_part17.hdf5',
#     batch_size=args.batch_size,
#     val_split=args.val_split,
#     subset_size=args.subset_size
# )

#     # initialise model, loss function, optimizer
#     #model = get_model(n_classes=6).to(device)
#     n_classes = 6  # number of output classes
#     ninp = 512  # embedding dimension (size of each input token)
#     nhead = 8  # number of heads in the multiheadattention models
#     nhid = 512  # dimension of the feedforward network model (hidden layer size)
#     nlayers = 2  # number of sub-encoder-layers in the transformer model
#     dropout = 0.2023326371316407  # dropout rate

#     model = get_transformer_model(n_classes, ninp, nhead, nhid, nlayers, dropout).to(device)
#     criterion = nn.BCEWithLogitsLoss().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=args.lr)

#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=2)
#     #scheduler = StepLR(optimizer, step_size=2, gamma=0.5) # controls learning rate

#     # store losses
#     train_losses = []
#     val_losses = []


#     print(f"Training set length: {len(train_loader.dataset)}")
#     print(f"Validation set length: {len(valid_loader.dataset)}")


#     # training loop
#     for epoch in range(args.epochs):
#         model.train()
#         running_loss = 0.0

#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(
#                 device, dtype=torch.float32)
#             batch_size, seq_len, features = inputs.shape

#             inputs = inputs.permute(1, 0, 2)
#             src_mask = model.generate_square_subsequent_mask(inputs.size(0)).to(device) #transformer modelling

#             optimizer.zero_grad()
#             with autocast():
#                 outputs = model(inputs, src_mask)
#                 loss = criterion(outputs, labels.float())

#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#             #loss.backward()
#             #optimizer.step()
            
#             running_loss += loss.item() * inputs.size(0)
        
#         # calculate and store epoch loss for training
#         epoch_loss = running_loss / len(train_loader.dataset)
#         scheduler.step(epoch_loss)
#         train_losses.append(epoch_loss)
#         print(f'Epoch {epoch + 1}/{args.epochs}, Loss: {epoch_loss:.4f}')

#         # validation loop
#         model.eval()
#         val_running_loss = 0.0
#         with torch.no_grad(), autocast():
#             for inputs, labels in valid_loader:
#                 inputs = inputs.to(device, dtype=torch.float32)
#                 inputs = inputs.permute(1, 0, 2)  # Reshape for the Transformer
#                 src_mask = model.generate_square_subsequent_mask(inputs.size(0)).to(device)
        
#                 labels = labels.to(device, dtype=torch.float32)

#                 outputs = model(inputs, src_mask)
#                 loss = criterion(outputs, labels.float())
#                 val_running_loss += loss.item() * inputs.size(0)

#         # calculate and store epoch loss for validation
#         val_epoch_loss = val_running_loss / len(valid_loader.dataset)
#         scheduler.step(val_epoch_loss)
#         val_losses.append(val_epoch_loss)
#         print(f'Validation Loss: {val_epoch_loss:.4f}')

#     # Create a string of hyperparameters
#     hyperparameters_str = "\nHyperparameters used for training:\n"
#     hyperparameters_str += f"Path to HDF5 dataset: {args.path_to_hdf5}\n"
#     hyperparameters_str += f"Path to CSV annotations: {args.path_to_csv}\n"
#     hyperparameters_str += f"Validation split ratio: {args.val_split}\n"
#     hyperparameters_str += f"Batch size: {args.batch_size}\n"
#     hyperparameters_str += f"Number of epochs: {args.epochs}\n"
#     hyperparameters_str += f"Learning rate: {args.lr}\n"
#     hyperparameters_str += f"n_classes : {n_classes}\n"
#     hyperparameters_str += f"ninp: {ninp}\n"
#     hyperparameters_str += f"nhid: {nhid}\n"
#     hyperparameters_str += f"n_layers: {nlayers}\n"
#     hyperparameters_str += f"dropout: {dropout}\n"

#     if args.subset_size is not None:
#         hyperparameters_str += f"Subset size of data: {args.subset_size}\n"
#     else:
#         hyperparameters_str += "Subset size of data: Using all available data\n"
    
#     print(hyperparameters_str)
#     print("End of Training.")
#     torch.save(model.state_dict(), 'final_model.pth')

#     # plot training and validation loss
#     plt.figure(figsize=(10, 5))
#     plt.plot(train_losses, label='Training Loss')
#     plt.plot(val_losses, label='Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Loss over Epochs')
#     plt.legend()
#     plt.show()
