# BSD-3-Clause license

import torch
import torch.nn as nn
import torch.nn.functional as F
from engine.seg_losses import multi_loss_fusion,  multi_loss_fusion_kl

class REBNCONV(nn.Module):
    
    def __init__(self, in_ch = 3, out_ch = 3, dirate = 1):
        super(REBNCONV, self).__init__()
        
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding = 1 * dirate, dilation = 1 * dirate) #padding = 1 * dirate to keep the spatial dim same
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
        
        
        #Spatial resolution of any block in the RSU-L block remains indentical to the input feature map.
        #This is done by : padding = 1 * dirate
        
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
        hx1 = self.rebnconv1(hxin)   #RSU: local + multi scale features 
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

      

### RSU-6 ####
class RSU6(nn.Module):
    
    def __init__(self, in_ch = 3, mid_ch = 12, out_ch = 3):
        super(RSU6, self).__init__()
        
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate = 1)
        
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)   
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride = 2, ceil_mode=True)
        
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride = 2, ceil_mode=True)
        
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)        
        
        #Decoder
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)
        
    
    def forward(self, x):
        
        hx = x
        
        hxin = self.rebnconvin(x)
        
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        
        
        hx5 = self.rebnconv5(hx)
        
        hx6 = self.rebnconv6(hx5)
        
        
        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)
        
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup  = _upsample_like(hx4d, hx3)
        
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        
        
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        
        return hx1d + hxin
        
        
        
        
        
class RSU5(nn.Module):
    
    def __init__(self, in_ch = 3, mid_ch = 12, out_ch = 3):
        super(RSU5, self).__init__()
        
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate = 1)
        
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride = 2, ceil_mode=True)  
        
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride = 2, ceil_mode=True)   
        
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)
        
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1) 
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconvv1d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        
    
    def forward(self, x):
        
        hx = x
        
        hxin = self.rebnconvin(x)
        
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        
        hx4 = self.rebnconv4(hx)
        
        hx5 = self.rebnconv5(hx)


        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3) 
        
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        
        return hx1d + hxin
    
    
    
class RSU4(nn.Module):
    
    def __init__(self, in_ch = 3, mid_ch = 12, out_ch = 3):
        super(RSU5, self).__init__()
        
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate = 1)
        
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride = 2, ceil_mode=True)  
        
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)
        
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1) 
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        
    
    def forward(self, x):
        
        hx = x
        
        hxin = self.rebnconvin(x)
        
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        
        hx3 = self.rebnconv3(hx)
        
        hx4 = self.rebnconv4(hx)
        
        
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        
        return hx1d + hxin

   

### RSU-4F #####
class RSU4F(nn.Module):
    
    def __init__(self, in_ch = 3, mid_ch = 12, out_ch = 3):
        super(RSU4F, self).__init__()
        
        
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate = 1)
        
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1) 
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconvv1d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        
    
    def forward(self, x):
        
        hx = x
        
        hxin = self.rebnconvin(x)
        
        hx1 = self.rebnconv1(hxin)
        
        hx2 = self.rebnconv2(hx)
        
        hx3 = self.rebnconv3(hx)
        
        hx4 = self.rebnconv4(hx)
        
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))
        
        return hx1d + hxin



#### *************** ISNet ******  #######
class myrebnconv(nn.Module):
    def __init__(self, in_ch = 3, out_ch = 1, kernel_size = 1, stride = 1, padding = 1, dilation = 1, groups = 1):
        super(myrebnconv, self).__init__()
        
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, groups)
        self.bn = nn.ReLU(inplace = True)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        
        return self.relu(self.bn(self.conv(x)))




class ISNetGTEncoder(nn.Module):
    
    def __init__(self, in_ch = 1, out_ch = 1):
        super(ISNetGTEncoder, self).__init__()
        
        self.conv_in = myrebnconv(in_ch, 16, 3, stride=2, padding=1)
     
        
        #only encoder
        self.stage1 = RSU7(16, 16, 64)
        self.pool12 = nn.MaxPool2d(2, stride = 2, ceil_mode=True) #ceil_mode = True ensure that the extra partially covered regions is included with black padded pixels if necessary
        
        self.stage2 = RSU6(64, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage3 = RSU5(64, 32, 128)
        self.pool34 = nn.MaxPool2d(2, stride = 2, ceil_mode=True)
        
        self.stage4 = RSU4(128, 32, 256)
        self.pool45 = nn.MaxPool2d(2, stride = 2, ceil_mode=True)
        
        self.stage5 = RSU4F(256, 64, 256)
        self.pool56 = nn.MaxPool2d(2, stride = 2, ceil_mode=True)
        
        self.stage6 = RSU4F(512, 64, 512)
    
        
        self.side1 = nn.Conv2d(64,  out_ch ,3, padding=1)  #3x3 convolutions
        self.side2 = nn.Conv2d(64,  out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding = 1)
        

    def compute_loss(self, preds, targets):
        
        return multi_loss_fusion(preds, targets)
        
    def forward(self, x):
        
        hx = x
        
        hxin = self.conv_in(hx)
        
        #stage1
        hx1 = self.stage1(hxin)
        hx = self.pool12(hx1)
        
        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        
        #stage3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        
        
        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        
        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        
        #stage 6
        hx6 = self.stage6(hx)
     
        
        #side output
        d1 = self.side1(hx1)
        
        d2 = self.side2(hx2)
        d2 = _upsample_like(d2, x)
        
        d3 = self.side3(hx3)
        d3 = _upsample_like(d3, x)
        
        d4 = self.side4(hx4)
        d4 = _upsample_like(d4, x)
        
        d5 = self.side5(hx5)
        d5 = _upsample_like(d5, x)
        
        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, x)
        
        # d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6),1))
        
        return  F.sigmoid(d1), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6), [hx1, hx2, hx3, hx4, hx5, hx6]


# *************************************************************

class ISNetDIS(nn.Module):
    
    def __init__(self, in_ch = 3, out_ch = 1):
        super(ISNetDIS, self).__init__()
        
        self.conv_in = nn.Conv2d(in_ch, 64, 3, stride=2, padding=1)
        self.pool_in = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage1 = RSU7(64, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride = 2, ceil_mode=True) #ceil_mode = True ensure that the extra partially covered regions is included with black padded pixels if necessary
        
        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride = 2, ceil_mode=True)
        
        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride = 2, ceil_mode=True)
        
        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride = 2, ceil_mode=True)
        
        self.stage6 = RSU4F(512, 256, 512)
        
        #decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)
        
        
        self.side1 = nn.Conv2d(64, out_ch , 3, padding=1)  #3x3 convolutions
        self.side2 = nn.Conv2d(64, out_ch, 3 , padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3,  padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding = 1)
        
        # self.outconv = nn.Con2d(6*out_ch, out_ch, 1) #1x1 convolution
     
    def compute_loss_kl(self, preds, targets, dfs, fs, mode = 'MSE'):
        
        return multi_loss_fusion_kl(preds, targets, dfs, fs, mode = mode)   

    def compute_loss(self, preds, targets):
        
        return multi_loss_fusion(preds, targets)
        
    def forward(self, x):
        
        hx = x
        
        hxin = self.conv_in(hx)
        
        #stage1
        hx1 = self.stage1(hxin)
        hx = self.pool12(hx1)
        
        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        
        #stage3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        
        
        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        
        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        
        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)
        
        
        # ************** Decoder ****************************
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))
                
        
        #side output
        d1 = self.side1(hx1d)
        
        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)
        
        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)
        
        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)
        
        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)
        
        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)
        
        # d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6),1))
        
        return  F.sigmoid(d1), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6), [hx1d, hx2d, hx3d, hx4d, hx5d, hx6]