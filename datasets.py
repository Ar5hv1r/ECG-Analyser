from torch.utils.data import Dataset, DataLoader
import pandas as pd
import h5py
import numpy as np
import os

class ECGDataset(Dataset):
    def __init__(self, path_to_hdf5, path_to_csv, hdf5_filename="exams_part17.hdf5", subset_size=None, start_idx=0, end_idx=None):
        self.path_to_hdf5 = path_to_hdf5
        self.hdf5_filename = hdf5_filename # Specify the HDF5 filename you want to load
        self.labels_df = pd.read_csv(path_to_csv)
        # Filter the DataFrame to include only rows corresponding to the specified HDF5 file
        self.labels_df = self.labels_df[self.labels_df['trace_file'] == self.hdf5_filename]
        self.labels = self.labels_df[['1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF']].values.astype(float)

        if subset_size is not None:
            self.labels_df = self.labels_df.sample(n=subset_size, random_state=42)
            self.labels = self.labels_df[['1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF']].values.astype(float)
        

        if end_idx is None:
            end_idx = len(self.labels_df)

        self.start_idx = start_idx
        self.end_idx = end_idx

    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, idx):
        current_exam_id = self.labels_df.iloc[idx]['exam_id']
        
        # Use the specified HDF5 file instead of using the filename from the CSV
        full_hdf5_path = os.path.join(self.path_to_hdf5, self.hdf5_filename)
        
        with h5py.File(full_hdf5_path, "r") as hdf5_file:
            exam_ids = np.array(hdf5_file['exam_id'])
            tracings = np.array(hdf5_file['tracings'])
            
            exam_index = np.where(exam_ids == current_exam_id)[0][0]
            x = tracings[exam_index]
        
        y = self.labels[idx]
        #print(f'x: {len(x)}, y: {len(y)}')
        return x, y



# create DataLoader instances for training and validation sets

def get_train_and_val_loaders(path_to_hdf5, path_to_csv, hdf5_filename="exams_part17.hdf5", subset_size=100, batch_size=8, val_split=0.02):
    labels = pd.read_csv(path_to_csv)
    # Filter labels for the specific HDF5 file, if necessary
    labels = labels[labels['trace_file'] == hdf5_filename]
    n_samples = len(labels)
    print(f'Labels: {n_samples}')
    n_train = int(n_samples * (1 - val_split))
    print(f'n_train: {n_train}')

    train_dataset = ECGDataset(path_to_hdf5, path_to_csv, hdf5_filename=hdf5_filename, end_idx=n_train)
    valid_dataset = ECGDataset(path_to_hdf5, path_to_csv, hdf5_filename=hdf5_filename, start_idx=n_train)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, valid_loader
