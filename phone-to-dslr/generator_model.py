import torch
import torch.nn as nn

class genBlock(nn.Module):
    def __init__(self, in_channels, out_channels,  batch_norm = True, dropout=False, mode = 'Down', activation_type = 'relu'):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, padding_mode='reflect')
            if mode == 'Down'
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2 ,1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if activation_type == 'relu' else nn.LeakyReLU(0.2)
        )
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(0.5)

    def forward(self, x):
        output = self.block(x)
        if(self.dropout):
            return self.dropout_layer(output)
        else:
            return output

class Generator(nn.Module):
    def __init__(self,in_channels, features=64):
        super().__init__()
        self.encoders = [
        nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        ),
            genBlock(64, 128, activation_type ='leakyReLU'),  
            genBlock(128, 256, activation_type ='leakyReLU'),  
            genBlock(256, 512, activation_type ='leakyReLU'),  
            genBlock(512, 512, activation_type ='leakyReLU'),
            
        ]
        self.bottleneck = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), nn.ReLU())
        self.pre_decoder = genBlock(512, 512, mode = 'Up', dropout=True)
        self.decoders = [  
            genBlock(1024, 512, mode = 'Up', dropout=True),
            genBlock(1024, 256, mode = 'Up'),  
            genBlock(512, 128, mode = 'Up'),  
            genBlock(256, 64, mode = 'Up'), 

        ]
        self.final_conv = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, x):
        skips_cons = []
        for encoder in self.encoders:
            x = encoder(x)
            skips_cons.append(x)
        skips_cons = list(reversed(skips_cons))
        decoders = self.decoders
        x = self.bottleneck(x)
        i = 0
        x = self.pre_decoder(x)
        for decoder, skip in zip(decoders, skips_cons):
            x = torch.cat((x, skip), axis=1)
            x = decoder(x)

        x = self.final_conv(x)
        return self.tanh(x)
