import torch
import numpy as np
import argparse
from datasets import ECGDataset
from model import get_model
from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get performance on test set from hdf5")
    parser.add_argument('path_to_hdf5', type=str, help='path to hdf5 file containing tracings')
    parser.add_argument('path_to_model', type=str, help='path to the trained PyTorch model file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--output_file', type=str, required=True, help='File path to save the output predictions')  # Added argument for output file
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Assuming ECGDataset initialization without labels since only path_to_hdf5 is provided
    test_dataset = ECGDataset(args.path_to_hdf5, None)  # Adjust if your ECGDataset expects different arguments
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = get_model(n_classes=6)  # Number of classes needs to be adjusted based on your model requirements
    model.load_state_dict(torch.load(args.path_to_model, map_location=device))
    model.to(device)

    # Debugging line to check the device model parameters are loaded onto
    print(next(model.parameters()).device)

    model.eval()

    predictions = []
    with torch.no_grad():
        for inputs in test_loader:
            # Assuming inputs are directly from DataLoader without labels, adjust if your DataLoader returns a tuple
            inputs = inputs.to(device, dtype=torch.float32)
            outputs = model(inputs)
            # Sigmoid is applied to the outputs to turn logits into probabilities
            predictions.append(outputs.sigmoid().cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    np.save(args.output_file, predictions)  # Save the predictions to the specified output file
    print("Output predictions saved")
