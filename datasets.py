from torch.utils.data import Dataset, DataLoader
import pandas as pd
import h5py
import numpy as np
import os

class ECGDataset(Dataset):
    def __init__(self, path_to_hdf5, path_to_csv, start_idx=0, end_idx=None):
        """
        Initializes the ECG dataset.
        
        :param path_to_hdf5: Path to the directory containing HDF5 files.
        :param path_to_csv: Path to the CSV file containing metadata.
        :param subset_size: If specified, limits the number of records loaded.
        :param start_idx: Starting index for slicing the dataset.
        :param end_idx: Ending index for slicing the dataset.
        """
        self.path_to_hdf5 = path_to_hdf5
        all_labels = pd.read_csv(path_to_csv)

        self.labels = all_labels[all_labels['trace_file']=='exams_part17.hdf5']
        
        self.start_idx = start_idx
        self.end_idx = end_idx if end_idx is not None else len(self.labels)

    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, idx):
        """
        Retrieves the ECG tracing and corresponding labels for a given index.
        """
        # if idx < self.start_idx or idx >= self.end_idx:
        #     raise IndexError("Index out of bound")
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
        
        y = record[['1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF']].astype(int).values
        return tracing, y

def get_train_and_val_loaders(path_to_hdf5, path_to_csv, hdf5_filename, batch_size=8, val_split=0.02):
    """
    Prepares training and validation DataLoader instances.
    
    :param path_to_hdf5: Path to the directory containing HDF5 files.
    :param path_to_csv: Path to the CSV file containing metadata.
    :param hdf5_filename: The filename of the HDF5 file for filtering.
    :param batch_size: Batch size for the DataLoader.
    :param val_split: Ratio of the dataset to be used for validation.
    :param subset_size: If specified, limits the number of records loaded.
    """
    labels = pd.read_csv(path_to_csv)
    filtered_labels = labels[labels['trace_file'] == hdf5_filename]

    n_samples = len(filtered_labels)
    n_train = int(n_samples * (1 - val_split))

    train_dataset = ECGDataset(path_to_hdf5, path_to_csv, start_idx=0, end_idx=n_train)
    valid_dataset = ECGDataset(path_to_hdf5, path_to_csv, start_idx=n_train, end_idx=n_samples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader
