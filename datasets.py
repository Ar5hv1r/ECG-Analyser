from torch.utils.data import Dataset, DataLoader
import pandas as pd
import h5py
import numpy as np

# custom dataset class for ECG data


class ECGDataset(Dataset):
    def __init__(self, path_to_hdf5, path_to_csv, hdf5_dset="tracings", start_idx=0, end_idx=None):
        self.path_to_hdf5 = path_to_hdf5
        self.hdf5_dset = hdf5_dset
        self.labels = pd.read_csv(
            path_to_csv).values if path_to_csv is not None else None

        # open .hdf5 file and load dataset
        with h5py.File(self.path_to_hdf5, "r") as f:
            self.data = f[self.hdf5_dset]
            if end_idx is None:
                end_idx = len(self.data)

        self.start_idx = start_idx
        self.end_idx = end_idx

    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, idx):
        idx = self.start_idx + idx
        with h5py.File(self.path_to_hdf5, "r") as f:
            x = np.array(f[self.hdf5_dset][idx]).astype(
                np.float32)  # cast to float32

        if self.labels is not None:
            y = np.array(self.labels[idx])
            return x, y
        else:
            return x

# create DataLoader instances for training and validation sets


def get_train_and_val_loaders(path_to_hdf5, path_to_csv, hdf5_dset="tracings", batch_size=8, val_split=0.02):
    labels = pd.read_csv(path_to_csv)
    n_samples = len(labels)
    n_train = int(n_samples * (1 - val_split))

    train_dataset = ECGDataset(
        path_to_hdf5, path_to_csv, hdf5_dset, end_idx=n_train)
    valid_dataset = ECGDataset(
        path_to_hdf5, path_to_csv, hdf5_dset, start_idx=n_train)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader
