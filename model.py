import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

#basic block
class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(ConvBlock, self).__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),
      nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU()
    )
    self.out_channels = out_channels
    
  def __call__(self, x):
    return self.conv(x)

class UpConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(UpConvBlock, self).__init__()
    self.conv = nn.Sequential(
      nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 3, padding = 1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),
      nn.ConvTranspose2d(out_channels, out_channels, kernel_size = 3, padding = 1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU()
    )
    self.out_channels = out_channels

  def __call__(self, x):
    return self.conv(x) 
  
class NaiveVAE(nn.Module):
  def __init__(self, in_channels = 1, out_channels = 1, init_features = 32, img_size = (384, 88)): 
    super(NaiveVAE, self).__init__()
    #encoder
    self.encoder1 = ConvBlock(in_channels, init_features)
    
    self.encoder2 = ConvBlock(init_features, init_features*2)
    
    self.encoder3 = ConvBlock(init_features*2, init_features*4)
    
    self.encoder4 = ConvBlock(init_features*4, init_features*8)
    
    #decoder
    self.decoder4 = UpConvBlock(init_features*8, init_features*4)
    
    self.decoder3 = UpConvBlock(init_features*4, init_features*2)
    
    self.decoder2 = UpConvBlock(init_features*2, init_features)
    
    self.output = UpConvBlock(init_features, out_channels)
    
  def forward(self, x):
    #encoder
    kernel_size = (2, 2)
    x = self.encoder1(x)
    
    x = self.encoder2(F.max_pool2d(x, kernel_size=kernel_size, stride=2))
    
    x = self.encoder3(F.max_pool2d(x, kernel_size=kernel_size, stride=2))

    x = self.encoder4(F.max_pool2d(x, kernel_size=kernel_size, stride=2))
    
    #decoder
    x = self.decoder4(F.interpolate(x, scale_factor=2, mode='bilinear'))

    x = self.decoder3(F.interpolate(x, scale_factor=2, mode='bilinear'))

    x = self.decoder2(F.interpolate(x, scale_factor=2, mode='bilinear'))

    x = self.output(x)
    return x
    
def get_model(img_size = (384, 88)):
  model = NaiveVAE(in_channels=1,out_channels=1,init_features=32, img_size = img_size)
  return model

if __name__ == "__main__":
    x = torch.rand(32, 1, 384, 88)
    model = get_model()
    output = model(x)
    print(output.shape)
