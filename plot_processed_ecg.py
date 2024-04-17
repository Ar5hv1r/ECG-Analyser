import torch
import matplotlib.pyplot as plt
from datasets import get_train_and_val_loaders
from model import get_transformer_model

def visualize_ecg_comparison(model, device, data_loader):
    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device, dtype=torch.float32)
            inputs_perm = inputs.permute(1, 0, 2)  # Adjust dimensions if necessary
            src_mask = model.generate_square_subsequent_mask(inputs_perm.size(0)).to(device)
            
            # Get the model's output
            outputs = model(inputs_perm, src_mask)
            
            # Convert inputs and outputs to CPU for plotting
            inputs_np = inputs.cpu().numpy()
            outputs_np = outputs.cpu().numpy()

            # Plotting the first example in the batch
            plt.figure(figsize=(14, 6))

            # Plot original ECG
            plt.subplot(1, 2, 1)
            plt.plot(inputs_np[0, :, :])  # Adjust indexing if necessary
            plt.title('Original ECG Signal')
            plt.xlabel('Time Points')
            plt.ylabel('Amplitude')
            plt.grid(True)

            # Plot processed ECG
            plt.subplot(1, 2, 2)
            plt.plot(outputs_np[0, :])  # Adjust indexing if necessary
            plt.title('Processed ECG Signal by Transformer Model')
            plt.xlabel('Time Points')
            plt.ylabel('Processed Signal Amplitude')
            plt.grid(True)

            plt.tight_layout()
            plt.show()
            
            break  # Only process the first batch for visualization

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
n_classes = 6
ninp = 512
nhead = 8
nhid = 2048
nlayers = 4
dropout = 0.5
model = get_transformer_model(n_classes, ninp, nhead, nhid, nlayers, dropout).to(device)
model.load_state_dict(torch.load('final_model.pth'))  # Load your model weights

# Prepare DataLoader
path_to_hdf5 = 'data'
path_to_csv = 'data/exams.csv'
hdf5_filename = 'exams_part17.hdf5'
train_loader, _ = get_train_and_val_loaders(path_to_hdf5, path_to_csv, hdf5_filename, batch_size=1, val_split=0.02)

# Visualize the processed ECG
visualize_ecg_comparison(model, device, train_loader)

# import torch
# import matplotlib.pyplot as plt
# from datasets import get_train_and_val_loaders
# from model import get_transformer_model

# def visualize_processed_ecg(model, device, data_loader):
#     model.eval()
#     with torch.no_grad():
#         for inputs, labels in data_loader:
#             inputs = inputs.to(device, dtype=torch.float32)
#             # Assuming the model expects inputs with dimensions [seq_len, batch_size, features]
#             inputs = inputs.permute(1, 0, 2)  # Adjust dimensions if necessary
#             src_mask = model.generate_square_subsequent_mask(inputs.size(0)).to(device)
            
#             # Get the model's output
#             outputs = model(inputs, src_mask)
            
#             # Plotting the first example in the batch
#             plt.figure(figsize=(12, 6))
#             plt.plot(inputs[:, 0, :].cpu())  # Plot the first channel of the first ECG in the batch
#             plt.title('Processed ECG Signal by Transformer Model')
#             plt.xlabel('Time Points')
#             plt.ylabel('Processed Signal Amplitude')
#             plt.grid(True)
#             plt.show()
            
#             break  # Only process the first batch for visualization

# # Set up device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load model
# n_classes = 6
# ninp = 512
# nhead = 8
# nhid = 2048
# nlayers = 4
# dropout = 0.5
# model = get_transformer_model(n_classes, ninp, nhead, nhid, nlayers, dropout).to(device)
# model.load_state_dict(torch.load('final_model.pth'))  # Load your model weights

# # Prepare DataLoader
# path_to_hdf5 = 'data'
# path_to_csv = 'data/exams.csv'
# hdf5_filename = 'exams_part17.hdf5'
# train_loader, _ = get_train_and_val_loaders(path_to_hdf5, path_to_csv, hdf5_filename, batch_size=1, val_split=0.02)

# # Visualize the processed ECG
# visualize_processed_ecg(model, device, train_loader)
