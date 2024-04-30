import torch
from model import TransformerModel

def load_model(model_path, n_classes=6, ninp=512, nhead=8, nhid=2048, nlayers=4, dropout=0.5):
    """
    Loads a pre-trained Transformer model from a specified path and configures it for evaluation.

    Args:
        model_path (str): The path to the trained model file
        n_classes (int): The number of output classes for the model.
        ninp (int): The number of input features or the embedding dimension size.
        nhead (int): The number of heads in the multi-head attention mechanism of the transformer.
        nhid (int): The dimension of the feedforward network model in transformer layers.
        nlayers (int): The number of sub-encoder layers in the transformer model.
        dropout (float): The dropout rate used in the transformer model.

    Returns:
        torch.nn.Module: The loaded and configured transformer model ready for inference.
    """
    # Choose the appropriate device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the TransformerModel with the specified parameters and move it to the selected device
    model = TransformerModel(ntoken=ninp, ninp=ninp, nhead=nhead, nhid=nhid, nlayers=nlayers, n_classes=n_classes, dropout=dropout).to(device)
    
    # Load the model state dictionary from the specified file path and ensure it is mapped to the correct device
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Set the model to evaluation mode, which turns off specific layers/functions like dropout and batch normalization during inference
    model.eval()

    return model

if __name__ == '__main__':
    model_path = 'final_model.pth'  # Define the path to the model file
    model = load_model(model_path)  # Load the model using the specified path
    print(model)  # Print the model's architecture and current state
