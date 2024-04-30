import torch
import numpy as np
import pandas as pd
import argparse
from datasets import get_train_and_val_loaders
from model import get_transformer_model
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def predict(model, device, test_loader):
    """
    Makes predictions using a trained model on data provided by a DataLoader.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        device (torch.device): The device (CPU or GPU) the model is running on.
        test_loader (DataLoader): DataLoader providing test dataset.

    Returns:
        tuple: A tuple containing an array of predictions and an array of true labels.
    """
    model.eval()  # Set the model to evaluation mode
    predictions = []
    labels_list = []
    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in test_loader:
            inputs = inputs.to(device, dtype=torch.float32).permute(1, 0, 2)  # Prepare inputs for the model
            src_mask = model.generate_square_subsequent_mask(inputs.size(0)).to(device)
            outputs = model(inputs, src_mask)  # Get model outputs
            predictions.append(outputs.sigmoid().cpu().numpy())  # Apply sigmoid activation and move to CPU
            labels_list.append(labels.cpu().numpy())  # Move labels to CPU
    predictions = np.concatenate(predictions, axis=0)  # Concatenate all predictions into one array
    labels = np.concatenate(labels_list, axis=0)  # Concatenate all labels into one array
    return predictions, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get predictions from hdf5")
    parser.add_argument('path_to_hdf5', type=str, help='Path to hdf5 file containing tracings')
    parser.add_argument('path_to_csv', type=str, help='Path to csv file containing annotations')
    parser.add_argument('path_to_model', type=str, help='Path to the trained PyTorch model file')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size for testing')
    parser.add_argument('--output_file', type=str, required=True, help='File path to save the output predictions')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Determine the computation device

    # Load the dataset using validation split
    _, test_loader = get_train_and_val_loaders(args.path_to_hdf5, args.path_to_csv, 'exams_part17.hdf5', batch_size=args.batch_size, val_split=0.2)
    
    # Initialize and load the trained model
    n_classes = 6  # Define the number of output classes
    ninp = 512  # Embedding dimension
    nhead = 8  # Number of heads in the multiheadattention models
    nhid = 512  # Dimension of the feedforward network model
    nlayers = 4  # Number of sub-encoder-layers in the transformer model
    dropout = 0.10140566659044377  # Dropout rate
    model = get_transformer_model(n_classes, ninp, nhead, nhid, nlayers, dropout)
    model.load_state_dict(torch.load(args.path_to_model, map_location=device))
    model.to(device)

    # Perform predictions
    predictions, true_labels = predict(model, device, test_loader)

    # Save predictions
    np.save(args.output_file, predictions)
    print(f"Output predictions saved to {args.output_file}")

    # Compute and display evaluation metrics
    threshold = 0.5
    binary_predictions = (predictions > threshold).astype(int)  # Convert probabilities to binary predictions
    accuracy = accuracy_score(true_labels, binary_predictions)
    precision = precision_score(true_labels, binary_predictions, average='macro', zero_division=0)
    recall = recall_score(true_labels, binary_predictions, average='macro', zero_division=0)
    f1 = f1_score(true_labels, binary_predictions, average='macro', zero_division=0)

    # Generate and visualize the confusion matrix
    cm = confusion_matrix(true_labels.argmax(axis=1), binary_predictions.argmax(axis=1))
    class_labels = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF']
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')
    plt.savefig('Confusion Matrix.png')
    plt.show()

    # Output the metrics to a DataFrame for better readability
    df_metrics = pd.DataFrame({
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1-Score': [f1]
    })
    # Optionally save metrics to a CSV
    df_metrics.to_csv('metrics.csv', index=False)

    print("\nValidation Metrics:")
    print(df_metrics)
