import  torch
import torch.nn as nn   
import torch.nn.functional as F   




class BasicBlock(nn.Module):
    
    expansion = 1 # for basic block --> resnt18, 34, there won't be any expansion
    
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride=stride, padding=1, bias = False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias = False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.downsample = downsample
        
    
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = F.relu(out, inplace=True)
        return out
    

class BottleNeck:
    
    #default dilation
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(BottleNeck, self).__init__()  
      
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride=1, bias = False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias = False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion) 
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += identity 
        out = F.relu(out, inplace=True)
        
        return out
        
        
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 1000):
        super(ResNet, self).__init__()
        
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias = False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64,  blocks=layers[0])
        self.layer2 = self._make_layer(block, 128, blocks = layers[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, blocks = layers[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, blocks = layers[3], stride = 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(in_features= 512 * block.expansion, out_features=num_classes) #resnet18 --> 512, resnet50 --> 512*4 - 2048
        
        
    def _make_layer(self, block, out_channels, blocks, stride = 1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias = False),
                nn.BatchNorm2d(num_features= out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) #before layer1
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        
        return x
    

def resnet18():
    return ResNet(block = BasicBlock, layers = [2, 2, 2, 2]) #num_layers

def resnet34():
    return ResNet(block = BasicBlock, layers = [3, 4, 6, 3])

def resnet50():
    return ResNet(block = BottleNeck, layers = [3, 4, 6, 3])

def resnet101():
    return ResNet(block = BottleNeck, layers = [3, 4, 23, 3])

def resnet151():
    return ResNet(block = BottleNeck, layers = [3, 8, 36, 3])


        
        
        