# Transformer Model for ECG Classification

This project implements a Transformer-based model to classify electrocardiogram (ECG) signals. The repository includes scripts for training the model, loading datasets, making predictions, and evaluating the model's performance.

## Project Structure

- `train1.py`: Script for training the Transformer model with given hyperparameters and data.
- `datasets.py`: Contains the `ECGDataset` class and a function to load training and validation DataLoader instances.
- `model.py`: Defines the Transformer model architecture and the positional encoding.
- `predict.py`: A script to load a trained model and perform predictions on a new dataset.
- `load_model.py`: Utility script for loading a pre-trained model for evaluation or further training.
- `README.md`: Provides an overview and basic usage instructions for the project.

## Dependencies

This project requires the following libraries:
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

You can install these dependencies via pip:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model, use the `train1.py` script. You need to specify paths to the necessary HDF5 and CSV files containing the training data:

```bash
python train1.py <path_to_hdf5> <path_to_csv> --epochs 60 --batch_size 24
```

### Making Predictions

Use the `predict.py` script to make predictions using a trained model. Specify the necessary paths and the output file:

```bash
python predict.py <path_to_hdf5> <path_to_csv> <path_to_model> --batch_size 24 --output_file predictions.npy
```

### Loading and Evaluating the Model

To load and evaluate a pre-trained model, you can use the `load_model.py` script:

```bash
python load_model.py <path_to_trained_model>
```

## Configuration

Model configurations such as number of layers, hidden units, and dropout can be adjusted within the scripts before execution.

## Output

The training script will save the trained model and potentially plots of the training/validation loss. Prediction results will be saved to a specified NumPy file, and evaluation metrics will be printed to the console.

# Data

Data used in the study:

1. The `CODE` study cohort, with n=1,558,415 patients was used for training and testing:
   - exams from 15% of the patients in this cohort were used for testing. This sub-cohort is refered as `CODE-15%`. 
     The `CODE-15\%` dataset is openly available: [doi: 10.5281/zenodo.4916206 ](https://doi.org/10.5281/zenodo.4916206).



# References
This project was heavily inspired by the below paper

Scripts and modules for training and testing deep neural networks for ECG automatic classification.
Companion code to the paper "Deep neural network-estimated electrocardiographic age as a mortality predictor".
https://www.nature.com/articles/s41467-021-25351-7.

Citation:
```
Lima, E.M., Ribeiro, A.H., Paixão, G.M.M. et al. Deep neural network-estimated electrocardiographic age as a 
mortality predictor. Nat Commun 12, 5117 (2021). https://doi.org/10.1038/s41467-021-25351-7. 
```

Bibtex:
```bibtex
@article{lima_deep_2021,
  title = {Deep Neural Network Estimated Electrocardiographic-Age as a Mortality Predictor},
  author = {Lima, Emilly M. and Ribeiro, Ant{\^o}nio H. and Paix{\~a}o, Gabriela MM and Ribeiro, Manoel Horta and Filho, Marcelo M. Pinto and Gomes, Paulo R. and Oliveira, Derick M. and Sabino, Ester C. and Duncan, Bruce B. and Giatti, Luana and Barreto, Sandhi M. and Meira, Wagner and Sch{\"o}n, Thomas B. and Ribeiro, Antonio Luiz P.},
  year = {2021},
  journal = {Nature Communications},
  volume = {12},
  doi = {10.1038/s41467-021-25351-7},
  annotation = {medRxiv doi: 10.1101/2021.02.19.21251232},}
}
```
**OBS:** *The three first authors: Emilly M. Lima, Antônio H. Ribeiro, Gabriela M. M. Paixão contributed equally.*
