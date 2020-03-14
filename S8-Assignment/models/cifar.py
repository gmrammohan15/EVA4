import torch.nn as nn
import torch.nn.functional as F
dropout_value = 0.1

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1,  stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        ) #Output size=32 RF=1

        #Apply Depth wise separation(group)
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, groups=32, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        ) #Output size=32 RF=1

        self.pool1 = nn.MaxPool2d(2,2) #Output size=16 RF=1

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )  #Output size=16 RF=5

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )  #Output size=16 RF=9
        self.pool2 = nn.MaxPool2d(2,2)  #Output size=8 RF=11

        #Apply Dilation of 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=2, stride=1, dilation=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )  #Output size=8 RF=27

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )  #Output size=8 RF=35
        self.pool3 = nn.MaxPool2d(2,2)  #Output size=4 RF=39

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )  #Output size=4 RF=55


        #GAP Layer
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        )    #  RF = 79  

        #FC
        self.fc = nn.Linear(in_features=128,out_features=10)


    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x) 
        x = self.convblock4(x)
        
        x = self.pool2(x)    
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.pool3(x) 
        x = self.convblock7(x)
        
        x = self.gap(x)  
        x = x.view(-1,128)
        x = self.fc(x)      
         
        return x

    def summary(self, input_size):
      summary(self, input_size=input_size)

    def evaluate(self, optimizer, train_loader, test_loader, epochs, statspath, scheduler=None, batch_scheduler=False, l1_lambda=0):
      self.trainer = ModelTrainer(self, optimizer, train_loader, test_loader, statspath, scheduler, batch_scheduler, l1_lambda)
      self.trainer.run(epochs)

    def stats(self):
      return self.trainer.stats if self.trainer else None