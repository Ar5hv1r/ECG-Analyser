from torch.utils.data import Dataset, DataLoader
import pandas as pd
import h5py
import numpy as np
import os

class ECGDataset(Dataset):
    """
    A PyTorch Dataset class for loading ECG data from HDF5 files based on metadata in CSV format.
    """
    def __init__(self, path_to_hdf5, path_to_csv, start_idx=0, end_idx=None, subset_size=None):
        """
        Initializes the dataset by specifying the location of HDF5 and CSV files, the index range, and the subset size.

        Args:
            path_to_hdf5 (str): Path to the directory containing HDF5 files.
            path_to_csv (str): Path to the CSV file containing metadata.
            start_idx (int): Starting index for slicing the dataset. Default is 0.
            end_idx (int, optional): Ending index for slicing the dataset. If None, the entire length is considered.
            subset_size (int, optional): If specified, randomly selects a subset of the data for faster loading.
        """
        self.path_to_hdf5 = path_to_hdf5
        all_labels = pd.read_csv(path_to_csv)

        self.labels = all_labels[all_labels['trace_file']=='exams_part17.hdf5']
        
        if subset_size is not None:
            self.labels = self.labels.sample(n=subset_size, random_state=42)  # Subsampling with random state for reproducibility

        self.start_idx = start_idx
        self.end_idx = end_idx if end_idx is not None else len(self.labels)

    def __len__(self):
        """Returns the number of items in the dataset."""
        return self.end_idx - self.start_idx

    def __getitem__(self, idx):
        """
        Retrieves the ECG tracing and corresponding labels for a given index.
        
        Args:
            idx (int): Index of the data point to retrieve.

        Returns:
            tuple: A tuple containing the normalized ECG tracing data and its corresponding label vector.

        Raises:
            IndexError: If the index is out of the defined bounds.
        """
        if idx < 0 or idx >= len(self.labels):
            print(f"Requested idx: {idx}, but dataset size is: {len(self.labels)}")
            raise IndexError("Index out of bound")
        
        actual_idx = idx + self.start_idx
        record = self.labels.iloc[actual_idx]
        hdf5_file_name = record['trace_file']
        exam_id = record['exam_id']

        with h5py.File(os.path.join(self.path_to_hdf5, hdf5_file_name), 'r') as hdf5_file:
            exam_ids = np.array(hdf5_file['exam_id'])
            exam_index = np.where(exam_ids == exam_id)[0][0]
            tracing = hdf5_file['tracings'][exam_index, :, :]
        
        # Z-score Normalization: normalize data to have mean=0 and std=1
        mean = tracing.mean(axis=0)
        std = tracing.std(axis=0)
        std[std == 0] = 1  # Prevent division by zero
        tracing = (tracing - mean) / std

        # Min-Max Normalization with checks for division by zero
        # range = tracing.max(axis=0) - tracing.min(axis=0)
        # range[range == 0] = 1  # Avoid division by zero by setting zero ranges to 1
        # tracing = (tracing - tracing.min(axis=0)) / range

        y = record[['1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF']].astype(int).values
        return tracing, y

def get_train_and_val_loaders(path_to_hdf5, path_to_csv, hdf5_filename, batch_size=8, val_split=0.02, subset_size=None):
    """
    Prepares and returns DataLoader instances for both training and validation datasets.

    Args:
        path_to_hdf5 (str): Path to the directory containing HDF5 files.
        path_to_csv (str): Path to the CSV file containing metadata.
        hdf5_filename (str): The specific HDF5 file to use for filtering data.
        batch_size (int): Number of samples per batch.
        val_split (float): Fraction of the data to reserve for the validation set.
        subset_size (int, optional): If provided, limits the number of records loaded to a subset.

    Returns:
        tuple: A tuple containing DataLoader instances for both training and validation datasets.
    """
    labels = pd.read_csv(path_to_csv)
    filtered_labels = labels[labels['trace_file'] == hdf5_filename]

    if subset_size is not None:
        filtered_labels = filtered_labels.sample(n=subset_size, random_state=42)  # Sampling for reproducibility

    n_samples = len(filtered_labels)
    n_train = int(n_samples * (1 - val_split))

    train_dataset = ECGDataset(path_to_hdf5, path_to_csv, start_idx=0, end_idx=n_train, subset_size=subset_size)
    valid_dataset = ECGDataset(path_to_hdf5, path_to_csv, start_idx=n_train, end_idx=n_samples, subset_size=subset_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

    return train_loader, valid_loader
