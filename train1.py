import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
from datasets import get_train_and_val_loaders
from model import get_model
import matplotlib.pyplot as plt

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # if using multi-GPU.
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train neural network')
    # define command line args
    parser.add_argument('path_to_hdf5', type=str,
                        help='path to hdf5 file containing tracings')
    parser.add_argument('path_to_csv', type=str,
                        help='path to csv file containing annotations')
    parser.add_argument('--val_split', type=float,
                        default=0.02, help='validation split ratio')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='training batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    args = parser.parse_args()

    # set up device (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader = get_train_and_val_loaders(
        args.path_to_hdf5, args.path_to_csv, val_split=args.val_split, batch_size=args.batch_size)

    # initialise model, loss function, optimizer
    model = get_model(n_classes=6).to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=10, verbose=True)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1) # controls learning rate

    # store losses
    train_losses = []
    val_losses = []

    # training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(
                device, dtype=torch.float32)
            # debug
            # print(f"Train Loop - Inputs device: {inputs.device}, Labels device: {labels.device}, Model device: {next(model.parameters()).device}")
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)

        # calculate and store epoch loss for training
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f'Epoch {epoch + 1}/{args.epochs}, Loss: {epoch_loss:.4f}')

        # validation loop
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.float32)
                # debug
                # print(f"Validation Loop - Inputs device: {inputs.device}, Labels device: {labels.device}, Model device: {next(model.parameters()).device}")
                outputs = model(inputs)
                loss = criterion(outputs, labels.float())
                val_running_loss += loss.item() * inputs.size(0)

        # calculate and store epoch loss for validation
        val_epoch_loss = val_running_loss / len(valid_loader.dataset)
        scheduler.step(val_epoch_loss)
        val_losses.append(val_epoch_loss)
        print(f'Validation Loss: {val_epoch_loss:.4f}')

    print("End of Training.")
    torch.save(model.state_dict(), 'final_model.pth')

    # plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.show()
