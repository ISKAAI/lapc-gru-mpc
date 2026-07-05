import torch
import torch.nn as nn

class DisturbanceGRU(nn.Module):
    def __init__(self, n_feat=6, hidden=32, horizon=5):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, batch_first=True)
        self.head = nn.Linear(hidden,horizon)

    def forward(self,x):
        out, _ = self.gru(x)
        last = out[:,-1,:]
        return self.head(last)

if __name__ == "__main__":
    model = DisturbanceGRU()
    x = torch.randn(4,20,6)
    y = model(x)
    print(y.shape)