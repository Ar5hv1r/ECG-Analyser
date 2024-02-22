import h5py
import matplotlib.pyplot as plt

def plot_ecg_tracing(file_path, recording_index=0, channel_index=1):
    try:
        with h5py.File(file_path, 'r') as file:
            ecg_data = file['tracings'][recording_index][:, channel_index]

        plt.figure(figsize=(10,4))
        plt.plot(ecg_data)
        plt.title(f"ECG Tracing - Recording {recording_index+1}, Channel {channel_index+1}")
        plt.xlabel("Time Points")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()
    except KeyError:
        print("Specified dataset 'tracings' not found in the file")
    except IndexError:
        print("Specified recording_index or channel_index is out of bounds")
    except Exception as e:
        print(f"An error occurred: {e}")

file_path = 'data/ecg_tracings.hdf5'
plot_ecg_tracing(file_path, 0, 1)
