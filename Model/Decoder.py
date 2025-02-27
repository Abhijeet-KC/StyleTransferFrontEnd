import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    
    def __init__(self, d_model=768, seq_input=False, final_channels=3):  
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.seq_input = seq_input
        
        self.decoder = nn.Sequential(
            # Upsample Layer 1: 32x32 -> 64x64, 768 -> 256
            nn.ReflectionPad2d(1),
            nn.Conv2d(d_model, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # Additional conv layers at this size
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),

            # Upsample Layer 2: 64x64 -> 128x128, 256 -> 128
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            # Additional conv layer at this size
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            
            # Upsample Layer 3: 128x128 -> 256x256, 128 -> 64
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # Additional conv layer at this size
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            
            # Upsample Layer 4: 256x256 -> 256x256, 64 -> final_channels
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, final_channels, kernel_size=3, stride=1, padding=0),
        )
    
    def forward(self, x, input_resolution):
        if self.seq_input == True:
            B, N, C = x.size()
            (H, W) = input_resolution
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.decoder(x) 
            x = nn.Tanh()(x) 
            return x

# Testing
if __name__ == "__main__":
    model = Decoder(d_model=768, seq_input=True)
    x = torch.randn(2, 1024, 768)  # Simulating input
    output = model(x, input_resolution=(32, 32))
    print(output.shape)  # Should match expected output