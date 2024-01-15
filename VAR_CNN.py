import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class VarCNN(nn.Module):
    def __init__(self):
        super(VarCNN,self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=5,padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride =2),
            nn.Conv2d(64,64,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(64,64,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(64,64,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.dilation_blocks = nn.Sequential(
            nn.Conv2d(64,32,kernel_size=3,padding=2,dilation=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(32,16,kernel_size=3,padding=2,dilation=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(16,16,kernel_size=3,padding=2,dilation=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.dense_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16,400),
            nn.ReLU(),
            nn.Linear(400,512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512,400),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(400,1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv_blocks(x)
        x = self.dilation_blocks(x)
        x = self.dense_layers(x)
        x = self.sigmoid(x)
        return x