import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class token_acc_dataset(Dataset):
    def __init__(self, data, mean, std):
        self.X = (data[['cand_entropy', 'cand_prob']].values - mean) / std
        self.y = data['accepted'].values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)
    
class decision_model(nn.Module):
    def __init__(self):
        super(decision_model, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)
    
def test_model(test_loader, model, weight_path='/home/iasl-transformers/UCI-IASL-Transformer/decision_model/decision_model.pth'):
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    correct = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs).squeeze()
            # print(outputs)
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()

    accuracy = correct / len(test_loader.dataset)

    print(f"Test Accuracy: {accuracy}")

def prepare_data(file_path, test_size=0.2, val_size=0.1):
    data = pd.read_csv(file_path)
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)

    mean = test_data[['cand_entropy', 'cand_prob']].mean().values
    std = test_data[['cand_entropy', 'cand_prob']].std().values

    test_dataset = token_acc_dataset(test_data, mean, std)

    return test_dataset
if __name__ == "__main__":
    file_path = "/home/iasl-transformers/UCI-IASL-Transformer/decision_model/training_data_piqa.csv"
    test_dataset = prepare_data(file_path)
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model = decision_model().cuda()
    
    test_model(test_loader, model)
