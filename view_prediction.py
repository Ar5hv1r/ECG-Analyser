import numpy as np
import torch
from model import ECGNet
from torch.utils.data import DataLoader


# Assuming you have a dataset instance for training named 'train_dataset'
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# And for evaluation or testing, assuming you have 'test_dataset'
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


model = ECGNet(n_classes=6)
model.load_state_dict(torch.load('final_model.pth', map_location='cuda' if torch.cuda.is_available() else 'cpu'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def evaluate_accuracy(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode.
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():  # No need to calculate gradients
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # For binary classification, you might use a threshold, like 0.5, to determine the predicted class
            # predictions = (outputs.sigmoid() > 0.5).long()  # Assuming binary classification and outputs are logits
            
            # For multi-class classification, you find the class with the highest output value
            _, predictions = torch.max(outputs, 1)
            
            total_predictions += labels.size(0)
            correct_predictions += (predictions == labels).sum().item()

    accuracy = correct_predictions / total_predictions
    return accuracy
model.load_state_dict(torch.load('final_model.pth', map_location=device))
model.to(device)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)
test_accuracy = evaluate_accuracy(model, test_loader, device)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

validation_accuracy = evaluate_accuracy(model, validation_loader, device)
print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")
