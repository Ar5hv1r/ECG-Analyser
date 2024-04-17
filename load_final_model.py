import torch
from model import TransformerModel

def load_model(model_path, n_classes=6, ninp=512, nhead=8, nhid=2048, nlayers=4, dropout=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the TransformerModel with the correct parameters
    model = TransformerModel(ntoken=ninp, ninp=ninp, nhead=nhead, nhid=nhid, nlayers=nlayers, n_classes=n_classes, dropout=dropout).to(device)
    
    # Load the model state dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Set model to evaluation mode
    model.eval()

    return model

if __name__ == '__main__':
    model_path = 'final_model.pth'
    model = load_model(model_path)
    print(model)
