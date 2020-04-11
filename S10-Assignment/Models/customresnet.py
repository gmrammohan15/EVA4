import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1)
    
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(residual)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()

        block = BasicBlock
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)


        # self.convy = nn.Conv2d(64, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bny = nn.BatchNorm2d(self.inplanes)
        # self.maxpooly = nn.MaxPool2d(2,2)
        # self.reluy = nn.ReLU(inplace=True)

        self.convx = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnx = nn.BatchNorm2d(256)
        self.maxpoolx = nn.MaxPool2d(2,2)
        self.relux = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.layer1 = self._make_layer(block, 128, 1)
        self.layer2 = self._make_layer(block, 512, 1, inplanes=256)

        self.maxpool4 = nn.MaxPool2d(kernel_size=4, stride=1, padding=0)

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(512 , 10)

    def _make_layer(self, block, planes, blocks, stride=1, inplanes=64):
        downsample = None
        self.inplanes = inplanes
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes ,
                      kernel_size=3, padding=1, stride=stride, bias=False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        )

        layers = []

        
        self.inplanes = planes

        layers.append(block(self.inplanes, planes, stride, downsample))
        
        # for i in range(1, blocks):
        #     layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

      x = self.conv1(x)
      x = self.bn1(x)
      x = self.relu(x)
      

      # self.inplanes = 128

      # x = self.convy(x)
      # x = self.maxpooly(x)
      # x = self.bny(x)
      # x = self.reluy(x)

      x = self.layer1(x)

      self.inplanes = 256

      x = self.convx(x)
      x = self.maxpoolx(x)
      x = self.bnx(x)
      x = self.relux(x)

      self.inplanes = 256
      x = self.layer2(x)

      x = self.maxpool4(x)
      x = x.view(x.size(0), -1)
      x = self.fc(x)

      return F.log_softmax(x, dim=-1)