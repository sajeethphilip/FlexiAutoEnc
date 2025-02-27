import torch
import torch.nn as nn

class FlexiAutoencoder(nn.Module):
    def __init__(self):
        super(FlexiAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
        )

        # Decoder
        self.decoder = nn.Sequential(
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
