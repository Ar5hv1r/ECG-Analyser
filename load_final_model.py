import torch
from model import get_model

def load_model(model_path, n_classes=6):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(n_classes=n_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

if __name__ == '__main__':
    model_path = 'final_model.pth'
    model = load_model(model_path)
    print(model)