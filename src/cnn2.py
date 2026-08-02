"Superior Model of Convolutional Neural Network 1.0"

import torch as pt
from torch import nn

device = pt.accelerator.current_accelerator().type if pt.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class SCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear(x)
        return logits
    
model = SCNN().to(device)
#model = SCNN()
print(model.forward(pt.rand(1,28,28,device=device)))