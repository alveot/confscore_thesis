import os 
import transformers
import torch
from torch.utils.data import Dataset
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

class TextDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, usecols=["text", "label"])
        self.texts = self.data["text"]
        self.labels = self.data["label"]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded_text = model.encode([text])
        label = torch.tensor(self.labels[idx])
        one_hot = torch.nn.functional.one_hot(label, num_classes=2)
        return encoded_text[0], one_hot.float()


# Pytorch's nn module has lots of useful feature
import torch.nn as nn

class LSTMNet(nn.Module):
    def __init__(self):
        super(LSTMNet,self).__init__()
    
        # Dense layer to predict 
        self.fc = nn.Linear(384,2)
        # Prediction activation function
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,data):
        dense_outputs=self.fc(data)
        outputs=self.sigmoid(dense_outputs)
        return outputs
    

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = TextDataset(data_path="data/offenseval_train.csv")
    train_dataloader = DataLoader(train_dataset, batch_size=120)
    
    model = LSTMNet()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(),lr=1e-4)
    criterion = nn.BCELoss()
    criterion = criterion.to(device)
    
    for epoch in range(50):
        for batch in train_dataloader:
            data = batch[0]
            label = batch[1]
            predictions = model(data)
            loss = criterion(predictions, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    torch.save(model, "hate-classifier.pt")

            
if __name__ == '__main__':
    main()