import torch
import torch.nn as nn 
from torchvision.ops import Conv2dNormActivation
import torchvision.transforms.functional as F



class DoubleConv(nn.Module):
    def __init__(self, out_channels = 64):
        super().__init__()
        
        self.double_conv = nn.Sequential(
            Conv2dNormActivation(out_channels=out_channels, kernel_size=(3,3), stride=1, padding='same', bias = False),
            Conv2dNormActivation(out_channels = out_channels, kernel_size = (3, 3), stride = 1, padding = "same", bias  = False)
            
        )
    
    def forward(self, x):
        return self.double_conv(x)
    



class DecoderBlock(nn.Module):
    def __init__(self, out_channels = 64, kernel_size = 2, stride = 2):
        super().__init__()
        
        self.conv_transpose = nn.LazyConvTranspose2d(out_channels=out_channels, kernel_size=kernel_size,stride=stride)
        
        self.conv = DoubleConv(out_channels=out_channels)
    
    def forward(self, decoder_in, encoder_in):
        
        #Upsample feature map in the decoder path
        decoder_in = self.conv_transpose(decoder_in)
        
        #Resizing feature maps in the current decoder/expanding path to have the same shape
        #as the feature map obtained from the encoder/contracting layer
        
        decoder_in = F.resize(decoder_in, size=encoder_in.shape[2:])
        
        #Concatenating the feature map of same decoder and encoder level
        out = torch.cat((decoder_in, encoder_in), dim=1)
        
        #Information mixing and channel size reduction using Double Convolution
        out = self.conv(out)
        
        return out
    



class UNet(nn.Module):
    def __init__(self, num_classes: int = None):
        super().__init__()
        
        #Encoder Block 1
        self.enc_1 = DoubleConv(out_channels=64)
        
        #Encoder Block 2
        self.enc_2 = DoubleConv(out_channels=128)
        
        #Encoder Block 3
        self.enc_3 = DoubleConv(out_channels=256)
        
        #Encoder Block 4
        self.enc_4 = DoubleConv(out_channels=512)
        
        #Intermediate Block
        self.bottleneck = DoubleConv(out_channels=1024)
        
        #Decoder Block 1
        self.dec_1 = DecoderBlock(out_channels=1024)
        
        #Decoder Block 2
        self.dec_2 = DecoderBlock(out_channels=512)
        
        #Decoder Block 3
        self.dec_3 = DecoderBlock(out_channels=128)
        
        #Decoder Block 4
        self.dec_4 = DecoderBlock(out_channels=64)
        
        
        # 1x1 convolution to reduce the number of feature maps to number of classes
        self.conv_1x1 = nn.LazyConv2d(out_channels=num_classes, kernel_size=(1,1), stride=1, padding='same')
        
    
    def forward(self, x):
        
        #Encoder path
        enc_1 = self.enc_1(x)
        down_1 = F.max_pool2d(enc_1, kernel_size = 2)
        
        enc_2 = self.enc_2(down_1)
        down_2 = F.max_pool2d(enc_2, kernel_size = 2)
        
        enc_3 = self.enc_3(down_2)
        down_3 = F.max_pool2d(enc_3, kernel_size = 2)
        
        enc_4 = self.enc_4(down_3)
        down_4 = F.max_pool2d(enc_4, kernel_size = 2)
        
        #Intermediate
        bottleneck = self.bottleneck(down_4)
        
        #Decoder Path
        up_1 = self.dec_1(bottleneck, enc_4)
        
        up_2 = self.dec_2(up_1, enc_3)
        
        up_3 = self.dec_3(up_2, enc_2)
        
        up_4 = self.dec_4(up_3, enc_1)
        
        out = self.conv_1x1(up_4)
        
        
        return out
        
        