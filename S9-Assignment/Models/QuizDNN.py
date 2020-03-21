import torch.nn as nn
import torch.nn.functional as F

class DNNNet(nn.Module):
    def __init__(self):
        super(DNNNet, self).__init__()

        self.convInput = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 32, RF 3

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
        ) 

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
        ) 

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.convblockx4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), bias=False),
         ) 
    
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
        ) 
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        
        self.convblockx8 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), bias=False),
        ) 
        
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        ) 

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        ) 

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=8)
        ) # output_size = 1
        self.fc = nn.Linear(128, 10)


    def forward(self, x):

        x1 = self.convInput(x) 
        x2 = self.convblock1(x) 
        x = x1 + x2

        x3 = self.convblock2(x) 

        x = x + x3

        x = self.pool1(x) 


        x5 = self.convblockx4(x) 
        x6 = self.convblock4(x5) 
        x = x5 + x6

        x7 = self.convblock5(x) 

        x = x + x7

        x = self.pool2(x) 

        x9 = self.convblockx8(x)
        x10 = self.convblock6(x9)

        x = x9 + x10

        x11 = self.convblock7(x)

        x = self.gap(x11)
        x = x.view(-1, 128)
        x = self.fc(x)

        return x