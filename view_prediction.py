import numpy as np
import pandas as pd

predictions_file_path = 'prediction.npy'

# Load the predictions from the .npy file.
predictions = np.load(predictions_file_path)

# Print the shape of the predictions to understand its dimensions
print("Shape of predictions:", predictions.shape)

# Print the predictions
print("Predictions:\n", predictions)

# Print the first 10 predictions
print("First 10 predictions:\n", predictions[:10])

threshold = 0.5  # Define a threshold
binary_predictions = predictions > threshold  # Convert probabilities to binary outcomes based on the threshold

print("Binary predictions based on threshold of 0.5:\n", binary_predictions[:10])

class_labels = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF'] 

# Create a DataFrame for a nicer display
df_predictions = pd.DataFrame(predictions[:10], columns=class_labels)
df_binary_predictions = pd.DataFrame(binary_predictions[:10], columns=class_labels)

# Print out each prediction with its corresponding label
for i, sample_prediction in enumerate(predictions[:10]):  # Iterate over the first 10 samples
    print(f"Predictions for sample {i}:")
    for label, probability in zip(class_labels, sample_prediction):
        print(f"{label}: {probability:.4f}")  # Print each class probability with 4 decimal places
    print("-" * 30)  # Separator line for clarity

# Display the continuous and binary predictions in a tabular format for better understanding.
print("Continuous Predictions:")
print(df_predictions)
print("\nBinary Predictions (Threshold = 0.5):")
print(df_binary_predictions)
