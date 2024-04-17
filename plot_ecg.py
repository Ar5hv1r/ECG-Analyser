import h5py
import matplotlib.pyplot as plt
import pandas as pd

def plot_ecg_tracing(file_path, csv_path, recording_index=0, channel_index=1):
    try:
        # Load labels from CSV
        labels_df = pd.read_csv(csv_path)
        # Filter labels for part17 file only
        part17_labels = labels_df[labels_df['trace_file'] == 'exams_part17.hdf5']
        
        with h5py.File(file_path, 'r') as file:
            if 'tracings' not in file or 'exam_id' not in file:
                raise KeyError("Required datasets 'tracings' or 'exam_id' not found in the file.")

            ecg_data = file['tracings'][recording_index][:, channel_index]
            exam_ids = file['exam_id'][:]
            current_exam_id = exam_ids[recording_index]
            
            # Find labels and patient ID for the current exam_id
            current_labels = part17_labels[part17_labels['exam_id'] == current_exam_id]
            if current_labels.empty:
                raise ValueError("No matching labels or patient ID found for the current ECG tracing.")
            patient_id = current_labels.iloc[0]['patient_id']

        plt.figure(figsize=(10, 4))
        plt.plot(ecg_data)
        plt.title(f"ECG Tracing - Recording {recording_index + 1}, Channel {channel_index + 1}")
        plt.xlabel("Time Points")
        plt.ylabel("Amplitude")
        plt.grid(True)

        # Construct label string from the current label row
        label_names = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF']
        active_labels = [name for name in label_names if current_labels.iloc[0][name]]
        label_str = ', '.join(active_labels) if active_labels else "No labels active"
        
        # Adjust text positions and box attributes
        plt.figtext(0.5, 0.01, f"Labels: {label_str}", ha="center", fontsize=9, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 4})
        plt.figtext(0.25, 0.95, f"Exam ID: {current_exam_id}", ha="center", fontsize=9, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 4})
        plt.figtext(0.5, 0.95, f"Patient ID: {patient_id}", ha="center", fontsize=9, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 4})

        plt.show()
    except KeyError as e:
        print(e)
    except IndexError:
        print("Specified recording_index or channel_index is out of bounds")
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")

file_path = 'data/exams_part17.hdf5'
csv_path = 'data/exams.csv'
plot_ecg_tracing(file_path, csv_path, 1, 0)
