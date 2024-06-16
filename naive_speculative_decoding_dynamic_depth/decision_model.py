import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class token_acc_dataset(Dataset):
    def __init__(self, data, mean, std):
        self.X = (data[['cand_entropy', 'cand_prob']].values - mean) / std
        self.y = data['accepted'].values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

class BasicBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(BasicBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.relu = nn.ReLU()
        self.downsample = nn.Linear(in_features, out_features) if in_features != out_features else None

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out

class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        self.block1 = BasicBlock(2, 16)
        self.block2 = BasicBlock(16, 16)
        self.fc = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

def train_model(train_loader, val_loader, model, criterion, optimizer, epochs=10):
    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()

            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}")

def prepare_data(file_path, test_size=0.2, val_size=0.1):
    data = pd.read_csv(file_path)
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=42)

    mean = train_data[['cand_entropy', 'cand_prob']].mean().values
    std = train_data[['cand_entropy', 'cand_prob']].std().values

    train_dataset = token_acc_dataset(train_data, mean, std)
    val_dataset = token_acc_dataset(val_data, mean, std)
    test_dataset = token_acc_dataset(test_data, mean, std)

    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    file_path = "/home/iasl-transformers/UCI-IASL-Transformer/decision_model/training_data_combined.csv"
    train_dataset, val_dataset, test_dataset = prepare_data(file_path)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = ResNetModel().cuda()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(train_loader, val_loader, model, criterion, optimizer, epochs=10)
    torch.save(model.state_dict(), "resnet_model.pt")
