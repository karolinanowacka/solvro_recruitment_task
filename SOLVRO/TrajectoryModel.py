import torch
import torch.nn as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score

class TrajectoryModel(pl.LightningModule):
    def __init__(self):
        super(TrajectoryModel, self).__init__()
        self.conv1 = F.Conv1d(in_channels = 2, out_channels = 16, kernel_size = 5, stride = 1, padding = 2)
        self.conv2 = F.Conv1d(in_channels = 16, out_channels = 32, kernel_size = 5, stride = 1, padding = 2)
        self.conv3 = F.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 5, stride = 1, padding = 2)
        self.conv4 = F.Conv1d(in_channels = 64, out_channels = 128, kernel_size = 5, stride = 1, padding = 2)
        self.conv5 = F.Conv1d(in_channels = 128, out_channels = 256, kernel_size = 5, stride = 1, padding = 2)

        self.maxpool = F.MaxPool1d(kernel_size = 2, stride = 2)
        self.fc1 = F.Linear(256*150, 32) 
        self.fc2 = F.Linear(32, 5)

        self.dropout = F.Dropout(0.3)

        self.accuracy = Accuracy(task = 'multiclass', num_classes = 5)
        self.f1 = F1Score(task = 'multiclass', num_classes = 5)

    def forward(self, x):
        x = x.transpose(1,2)

        #print("shape after transpose: ", x.shape)
        x = self.conv1(x)
        #print("shape after conv1: ", x.shape)
        x = torch.relu(x)

        x = self.conv2(x)
        #print("shape after conv2: ", x.shape)
        x = torch.relu(x)

        x = self.conv3(x)
        #print("shape after conv3: ", x.shape)
        x = torch.relu(x)

        x = self.conv4(x)
        #print("shape after conv4: ", x.shape)
        x = torch.relu(x)

        x = self.conv5(x)
        #print("shape after conv5: ", x.shape)
        x = torch.relu(x)
        
        x = self.maxpool(x)
        x = self.dropout(x)
        
        #print("shape before flattening: ", x.shape)
        x = x.view(x.size(0), -1)
        #print("shape after flattening: ", x.shape)
        x = self.fc1(x)
        #print("shape after fc1: ", x.shape)
        
        x = torch.relu(x)

        x = self.fc2(x)
        #print("shape after fc2: ", x.shape)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.forward(x)
        loss = self.loss_fn(predictions, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.forward(x)
        loss = F.CrossEntropyLoss()(predictions, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 1e-3, weight_decay = 1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.1)
        return [optimizer], [scheduler]
    
    