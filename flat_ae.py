import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim=500*52, encoding_dim=1000, sensor_scales=[]):
        super(Autoencoder, self).__init__()

        self.sensor_scales = torch.tensor(sensor_scales, dtype=torch.float32)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 2),
            nn.ReLU(True),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.ReLU(True)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.ReLU(True),
            nn.Linear(encoding_dim * 2, input_dim),
            nn.Sigmoid() # Use Sigmoid to output values between 0 and 1
        )

    def forward(self, x):

        # print(f"x: {x}")
        encoded = self.encoder(x)
        # print(f"encoded:{encoded}")
        decoded = self.decoder(encoded)
        # print(f"decoded:{decoded}")

        return decoded