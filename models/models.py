import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricFCAutoencoder(nn.Module):
    def __init__(self, layer_sizes):
        super(SymmetricFCAutoencoder, self).__init__()
        sizes_len = len(layer_sizes)
        self.encoder = nn.Sequential(*[LinearWithReLU(layer_sizes[i], layer_sizes[i+1]) for i in range(sizes_len - 1)])
        decoder_layers = [LinearWithReLU(layer_sizes[-i], layer_sizes[-(i+1)]) for i in range(1, sizes_len - 1)]
        decoder_layers.append(nn.Linear(layer_sizes[1], layer_sizes[0]))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out

class LinearWithReLU(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearWithReLU, self).__init__()
        self.layers = nn.Sequential(nn.Linear(in_features, out_features),
                                    nn.ReLU())
    
    def forward(self, x):
        out = self.layers(x)
        return out

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoding Layers
        self.conv_1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv_2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pooling_func = nn.MaxPool2d(2, 2)

        # Decoding Layers
        self.trans_conv_1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.trans_conv_2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def forward(self, x):
        # Encode
        out = F.relu(self.conv_1(x))
        out = self.pooling_func(out)
        out = F.relu(self.conv_2(out))
        out = self.pooling_func(out)

        # Decode
        out = F.relu(self.trans_conv_1(out))
        out = torch.sigmoid(self.trans_conv_2(out))
        return out



if __name__ == '__main__':
    fcae = SymmetricFCAutoencoder((28*28, 32))
    print(fcae)
    convae = ConvAutoencoder()
    print(convae)
