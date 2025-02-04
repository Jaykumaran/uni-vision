import torch
import torch.nn as nn
import torch.nn.functional as F


class REBNCONV(nn.Module):
    
    def __init__(self, in_ch = 3, out_ch = 3, dirate = 1):
        super(REBNCONV, self).__init__()
        
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding = 1 * dirate, dilation = 1 * dirate): #padding = 1 * dirate to keep the spatial dim same
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))
        
        return xout
    

#upsample src to have the same spatial res as 'tar
def _upsample_like(src,tar):
    src = F.upsample(src, size = tar.shape[2:], mode = 'bilinear')
    return src

#For Explanation about U2Net : https://learnopencv.com/u2-net-image-segmentation/     
    
#RSU-7 #### 

### ******** MODIFIED RSU BLOCK **********************
### uses dilated convolutions to mitigate contextual information in deeper layers
class RSU7(nn.Module): #UNet07DRES
    
    def __init__(self, in_ch = 3, mid_ch = 12, out_ch = 3):
        super(RSU7, self).__init__()
        
        
        #Additional layer outside encoder for transformation
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        
        
        #******* ENCODER *********************
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride = 2, ceil_mode=True)
        
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        
        
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride = 2, ceil_mode=True)
        
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        
        
        self.rebnconv6 = REBNCONV(mid_ch,mid_ch, dirate=1)
        
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)
        
        
        #*********************** DECODER ***************
        
        self.rebnconv6d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        
    def forward(self, x):
        
        #intermediate features represented by : hx
        hx = x    #feature transformtion in general is represented by h(x)
        
        
        #Initial additional layer fwd pass
        hxin = self.rebnconvin(hx)
        
        #Encoder forward pass
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        
        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)
        
        hx6 = self.rebnconv6(hx)
        
        hx7 = self.rebnconv7(hx6)
        
        #Decoder forward pass
        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5)), 1)   
        hx5dup = _upsample_like(hx5d, hx4)
        
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4)), 1)
        hx4dup = _upsample_like(hx4d, hx3)
        
        
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3d)), 1)
        hx3dup = _upsample_like(hx3d, hx2)
        
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx3d)), 1)
        hx2dup = _upsample_like(hx2d, hx2d)
        
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1)) 
        
        return hx1d + hxin    
        
        
        
        
        
        
        
    
    