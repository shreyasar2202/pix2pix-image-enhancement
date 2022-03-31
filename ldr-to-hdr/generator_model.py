import torch
import torch.nn as nn

class genBlock(nn.Module):
    def __init__(self, in_channels, out_channels,  batch_norm = True, dropout=False, mode = 'Down', activation_type = 'relu'):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, padding_mode='reflect')
            if mode == 'Down'
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2 ,1),
            nn.InstanceNorm2d(out_channels),
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
#         self.final_conv2 = nn.Conv2d(6,3,kernel_size=1)
#         self.final_conv3 = nn.Conv2d(3,3,kernel_size=1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, x):
        x_input = x
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
#         x = torch.cat((x,x_input),dim=1)
#         x = self.final_conv2(x)
#         x = self.final_conv3(x)
        return self.relu(x)

# import torch
# import torch.nn as nn


# class Block(nn.Module):
#     def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
#         super(Block, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
#             if down
#             else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
#         )

#         self.use_dropout = use_dropout
#         self.dropout = nn.Dropout(0.5)
#         self.down = down

#     def forward(self, x):
#         x = self.conv(x)
#         return self.dropout(x) if self.use_dropout else x


# class Generator(nn.Module):
#     def __init__(self, in_channels=3, features=64):
#         super().__init__()
#         self.initial_down = nn.Sequential(
#             nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
#             nn.LeakyReLU(0.2),
#         )
#         self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
#         self.down2 = Block(
#             features * 2, features * 4, down=True, act="leaky", use_dropout=False
#         )
#         self.down3 = Block(
#             features * 4, features * 8, down=True, act="leaky", use_dropout=False
#         )
#         self.down4 = Block(
#             features * 8, features * 8, down=True, act="leaky", use_dropout=False
#         )
#         self.down5 = Block(
#             features * 8, features * 8, down=True, act="leaky", use_dropout=False
#         )

#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
#         )

#         self.up2 = Block(
#             features * 8 , features * 8, down=False, act="relu", use_dropout=True
#         )
#         self.up3 = Block(
#             features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
#         )
#         self.up4 = Block(
#             features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False
#         )
#         self.up5 = Block(
#             features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False
#         )
#         self.up6 = Block(
#             features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False
#         )
#         self.up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)
#         self.final_up = nn.Sequential(
#             nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
#             nn.Tanh(),
#         )

#     def forward(self, x):
#         d1 = self.initial_down(x)
#         d2 = self.down1(d1)
#         d3 = self.down2(d2)
#         d4 = self.down3(d3)
#         d5 = self.down4(d4)
#         d6 = self.down5(d5)
#         #d7 = self.down6(d6)
#         bottleneck = self.bottleneck(d6)

#         up2 = self.up2(bottleneck)
#         #up2 = self.up2(torch.cat([up1, d7], 1))
#         print(up2.shape, d6.shape)
#         up3 = self.up3(torch.cat([up2, d6], 1))
#         up4 = self.up4(torch.cat([up3, d5], 1))
#         up5 = self.up5(torch.cat([up4, d4], 1))
#         up6 = self.up6(torch.cat([up5, d3], 1))
#         up7 = self.up7(torch.cat([up6, d2], 1))
# #         print(d1.shape)
# #         print(d2.shape)
# #         print(d3.shape)
# #         print(d4.shape)
# #         print(d5.shape)
# #         print(d6.shape)
# #         print(bottleneck.shape)
# #         print()
# #         print(up2.shape)
# #         print(up3.shape)
# #         print(up4.shape)
# #         print(up5.shape)
# #         print(up6.shape)
# #         print(up7.shape)
#         return self.final_up(torch.cat([up7, d1], 1))


def test():
    x = torch.randn((1, 3, 256, 256))
    model = Generator(in_channels=3, features=64)
    preds = model(x)
    print(preds.shape)
    print(model)


if __name__ == "__main__":
    test()