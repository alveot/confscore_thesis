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
    
    test_dataset = TextDataset(data_path="data/offenseval_test.csv")
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    
    model_pretrained = torch.load('hate-classifier.pt').to('cpu')
    
    
    limit=0

    with torch.no_grad():
        for batch in test_dataloader:
            data = batch[0]
            label = batch[1]
            logits = model_pretrained(data)
            confscore=1-logits
            print("logits:{}, confscore:{}".format(logits,confscore))
            limit+=1
            if limit==500:
                break
if __name__ == '__main__':
    main()