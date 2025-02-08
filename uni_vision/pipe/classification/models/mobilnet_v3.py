import torch
import torch.nn as nn
import torch.nn.functional as F  



class HardSwish(nn.Module):
    def __init__(self, inplace = True):
        super(HardSwish, self).__init__()
        self.inplace = inplace
        
    def forward(self, x):
        return x * F.relu6(x + 3.0) / 6. #relu with a upper bound value
    
    

class HardSigmoid(nn.Module):
    def __init__(self, inplace = True):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace
    
    def forward(self, x):
        return F.relu6(x + 3., inplace = self.inplace) / 6.
    

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction = 4):
        super().__init__()
        
        squeeze_channels = in_channels // reduction # eg: 576 // 4 = 144
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        #fully covolutional network
        self.fc1 = nn.Conv2d(in_channels=in_channels, out_channels=squeeze_channels, kernel_size=1, stride=1)
        self.fc2 = nn.Conv2d(in_channels=squeeze_channels, out_channels=in_channels, kernel_size=1, stride=1)
        self.activation = nn.ReLU(inplace=True)
        self.scale_activation = HardSigmoid()
        
    def forward(self, x):
        scale = self.avgpool(x)
        scale = self.activation(self.fc1(scale))
        scale = self.scale_activation(self.fc2(scale))
        return x * scale
              
 
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion, use_se, activation):
        super().__init__()
        hidden_dim = int(round(in_channels * expansion))
        if in_channels == out_channels and stride == 1:
             self.use_residual = True
        else:
             self.use_residual = False
            
        layers = []
        
        if expansion != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride = 1, padding = 0, bias = False),
                nn.BatchNorm2d(hidden_dim, eps = 0.001, momentum = 0.01),
                activation()
            ])
        
        #if expansion = 1
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size,
                      stride = stride, padding = (kernel_size -1 ) // 2, groups = hidden_dim, bias = False),
            nn.BatchNorm2d(hidden_dim, eps = 0.001, momentum = 0.01),
            activation()
        ])
        
        #squeeze excitation isn't used in 3rd and 4th block of v3_small
        if use_se: #squeeze_excitation
            layers.append(SqueezeExcitation(hidden_dim))
        
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(out_channels, eps = 0.001, momentum = 0.01)
        ])
        
        self.block = nn.Sequential(*layers)
        
    
    def forward(self, x):
          
        if self.use_residual:
            return x + self.block(x)
        
        return self.block(x)
             

class MobileNetV3(nn.Module) :
    def __init__(self, num_classes = 1000, small = False):
        super().__init__()
        
        if small:
            mobilev3_config = [
                (16, 16, 3, 2, 1, True, nn.ReLU()),
                (16, 24, 3, 2, 72. / 16, False, nn.ReLU()),
                (24, 24, 3, 1, 88. / 24, False, nn.ReLU()),
                (24, 40, 5, 2, 4, True, HardSwish()),
                (40, 40, 5, 1, 6, True, HardSwish()),
                (40, 40, 5, 1, 6, True, HardSwish()),
                (40, 48, 5, 1, 3, True, HardSwish()),
                (48, 48, 5, 1, 3, True, HardSwish()),
                (48, 96, 5, 2, 6, True, HardSwish()),
                (96, 96, 5, 1, 6, True, HardSwish()),
                (96, 96, 5, 1, 6, True, HardSwish())
            ] 

            last_conv_channels = 576
            classifier_hidden = 1024
            stem_channels = 16
            
        else:
            mobilev3_config = [
                (16, 16, 3, 1, 1, False, nn.ReLU()),
                (16, 24, 3, 2, 4, False, nn.ReLU()),
                (24, 24, 3, 1, 3, False, nn.ReLU()),
                (24, 40, 5, 2, 3, True, nn.ReLU()),
                (40, 40, 5, 1, 3, True, nn.ReLU()),
                (40, 40, 5, 1, 3, True, nn.ReLU()),
                (40, 48, 5, 1, 3, True, nn.ReLU()),
                (48, 48, 5, 1, 3, True, nn.ReLU()),
                (48, 96, 5, 2, 6, True, HardSwish()),
                (96, 96, 5, 1, 6, True, HardSwish()),
                (96, 96, 5, 1, 6, True, HardSwish())
            ] 
            
            last_conv_channels = 960
            classifier_hidden = 1280
            stem_channels = 16
        
        #initial layer    
        layers = [
            nn.Conv2d(3, stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels, eps = 0.001, momentum = 0.01),
            HardSwish(inplace=True)
        ]
        
        for in_c, out_c, k, s, exp, se, act in mobilev3_config:
            layers.append(InvertedResidual(in_c, out_c, k, s, exp, se, act()))
        
        # in_channels from last layer of the inverted block #12th block in v3_small
        layers.append(nn.Conv2d(mobilev3_config[-1][1], last_conv_channels, kernel_size=1, stride = 1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(last_conv_channels, eps = 0.001, momentum = 0.01))
        layers.append(HardSwish())
        
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Sequential(
            nn.Linear(last_conv_channels, classifier_hidden),
            HardSwish(),
            nn.Dropout(0.2),
            nn.Linear(classifier_hidden, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.classifier(x)
        
        return x
        
        
        