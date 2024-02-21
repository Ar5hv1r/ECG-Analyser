import torch
import numpy as np
import argparse
from datasets import ECGDataset
from model import get_model
from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get performance on test set from hdf5")
    parser.add_argument('path_to_hdf5', type=str, help='path to hdf5 file containing tracings')
    parser.add_argument('path_to_model', help='path to the trained PyTorch model file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = ECGDataset(args.path_to_hdf5, None)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = get_model(n_classes=6) #adjustable
    model.load_state_dict(torch.load(args.path_to_model, map_location=device))
    model.to(device)

    #debug
    print(next(model.parameters()).device)

    model.eval()

    predictions = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device, dtype=torch.float32)
            outputs = model(inputs)
            predictions.append(outputs.sigmoid().cpu().numpy()) # can add sigmoid function

    predictions = np.concatenate(predictions, axis=0)
    np.save(args.output_file, predictions)
    print("Output predictions saved")