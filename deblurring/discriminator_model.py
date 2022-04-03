import torch
import torch.nn as nn
class discBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, batch_norm = True, kernel_size = 4):
        super().__init__()
        if(batch_norm):
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = 1, padding_mode ="reflect", bias = False),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.LeakyReLU(0.2)
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = 1, padding_mode ="reflect"),
                nn.LeakyReLU(0.2)
            )
        
    def forward(self, input):
        output = self.block(input)
        return output

class PatchGAN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.patchgan= nn.Sequential( 
        discBlock(in_channels*2, 64, 2, False),
        discBlock(64,128,2),
        discBlock(128,256,2),
        discBlock(256,512,1),
        nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
        )

    def forward(self, x, y):
        return self.patchgan(torch.cat([x,y], axis = 1))