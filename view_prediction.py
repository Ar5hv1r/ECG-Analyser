import numpy as np
import pandas as pd
# Replace 'path_to_predictions.npy' with the actual path to your .npy file
predictions_file_path = 'prediction.npy'

# Load the predictions
predictions = np.load(predictions_file_path)

# Print the shape of the predictions to understand its dimensions
print("Shape of predictions:", predictions.shape)

# Print the predictions
print("Predictions:\n", predictions)

# If you want to see a few predictions instead of all, you can slice the array:
# Print the first 5 predictions
print("First 5 predictions:\n", predictions[:10])

# If predictions are probabilities and you want to apply a threshold to get binary results:
threshold = 0.01  # Define a threshold
binary_predictions = predictions > threshold

print("Binary predictions based on threshold of 0.5:\n", binary_predictions[:10])

# If you have class labels and want to print predictions with labels for interpretation
class_labels = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF']  # Example class labels
# Create a DataFrame for a nicer display
df_predictions = pd.DataFrame(predictions[:10], columns=class_labels)
df_binary_predictions = pd.DataFrame(binary_predictions[:10], columns=class_labels)


# Print out each prediction with its corresponding label
for i, sample_prediction in enumerate(predictions[:10]):  # Iterate over the first 5 samples
    print(f"Predictions for sample {i}:")
    for label, probability in zip(class_labels, sample_prediction):
        print(f"{label}: {probability:.4f}")
    print("-" * 30)  # Separator line for clarity
print("Continuous Predictions:")
print(df_predictions)
print("\nBinary Predictions (Threshold = 0.5):")
print(df_binary_predictions)