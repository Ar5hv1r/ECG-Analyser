import torch
import numpy as np
import pandas as pd
import argparse
from datasets import ECGDataset, get_train_and_val_loaders
from model import get_transformer_model
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def predict(model, device, test_loader):
    model.eval()
    predictions = []
    labels_list = []
    with torch.no_grad():
        for inputs, labels in test_loader:  # Change made: DataLoader returns inputs and labels, but we discard labels (_)
            inputs = inputs.to(device, dtype=torch.float32).permute(1, 0, 2)  # Permute the inputs as the model expects
            src_mask = model.generate_square_subsequent_mask(inputs.size(0)).to(device)
            outputs = model(inputs, src_mask)  # forward pass
            predictions.append(outputs.sigmoid().cpu().numpy())  # apply sigmoid and move to cpu
            labels_list.append(labels.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels_list, axis=0)  # concatenate true labels
    return predictions, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get predictions from hdf5")
    parser.add_argument('path_to_hdf5', type=str, help='path to hdf5 file containing tracings')
    parser.add_argument('path_to_csv', type=str, help='path to csv file containing annotations')
    parser.add_argument('path_to_model', type=str, help='path to the trained PyTorch model file')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size for testing')
    parser.add_argument('--output_file', type=str, required=True, help='File path to save the output predictions')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    # Change made: Use the get_train_and_val_loaders but only get the validation part for prediction
    _, test_loader = get_train_and_val_loaders(args.path_to_hdf5, args.path_to_csv, 'exams_part17.hdf5', batch_size=args.batch_size, val_split=0.2)
    
    # Initialize and load the trained model
    n_classes = 6  # number of output classes
    ninp = 512  # embedding dimension (size of each input token)
    nhead = 8  # number of heads in the multiheadattention models
    nhid = 512  # dimension of the feedforward network model (hidden layer size)
    nlayers = 4  # number of sub-encoder-layers in the transformer model
    dropout = 	0.10140566659044377
    #dropout = 0.23537958833156938  # dropout rate
    model = get_transformer_model(n_classes, ninp, nhead, nhid, nlayers, dropout)
    model.load_state_dict(torch.load(args.path_to_model, map_location=device))
    model.to(device)

    # Make predictions
    predictions, true_labels = predict(model, device, test_loader)

    # Save the predictions to the specified output file
    np.save(args.output_file, predictions)
    print(f"Output predictions saved to {args.output_file}")
    # Calculate metrics using true labels and predicted probabilities
    # Define a threshold for binary classification
    threshold = 0.5
    binary_predictions = (predictions > threshold).astype(int)

    accuracy = accuracy_score(true_labels, binary_predictions)
    precision = precision_score(true_labels, binary_predictions, average='macro', zero_division=0)
    recall = recall_score(true_labels, binary_predictions, average='macro', zero_division=0)
    f1 = f1_score(true_labels, binary_predictions, average='macro', zero_division=0)

     # Generate and save confusion matrix
    cm = confusion_matrix(true_labels.argmax(axis=1), binary_predictions.argmax(axis=1))
    class_labels = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF']
    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')
    plt.savefig('Confusion Matrix.png')
    plt.show()

    # Create a DataFrame for displaying metrics
    df_metrics = pd.DataFrame({
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1-Score': [f1]
    })

    # Optionally, save the metrics to a CSV file
    df_metrics.to_csv('metrics.csv', index=False)

    print("\nValidation Metrics:")
    print(df_metrics)
