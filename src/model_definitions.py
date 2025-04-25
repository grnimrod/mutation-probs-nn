from torch import nn
import torch.nn.functional as F


class FullyConnectedNN(nn.Module):
        def __init__(self):
            super().__init__()

            self.linear_relu_seq = nn.Sequential(
                nn.LazyLinear(128),
                nn.ReLU(),
                # nn.Dropout(p=0.3),
                nn.LazyLinear(64),
                nn.ReLU(),
                # nn.Dropout(p=0.3),
                nn.LazyLinear(4)
            )
        
        def forward(self, x):
            x = self.linear_relu_seq(x)
            return x
        
        def predict_proba(self, x):
            logits = self.linear_relu_seq(x)
            return F.softmax(logits, dim=-1)