import numpy as np, torch,torch.nn as nn, os, sys
from torch.utils.data import TensorDataset, DataLoader
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from gru import DisturbanceGRU

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("metal")
else:
    device = torch.device("cpu")
    print("cpu")

data = np.load("processed/dataset.npz")
Xtr = torch.tensor(data["Xtr"]).float()
Ytr = torch.tensor(data["Ytr"]).float()
Xva = torch.tensor(data["Xva"]).float()
Yva = torch.tensor(data["Yva"]).float()

train_loader = DataLoader(TensorDataset(Xtr,Ytr), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(Xva,Yva),batch_size=64)

model = DisturbanceGRU()
model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay= 1e-4)

train_losses, val_losses = [], []
for epoch in range(50):
    model.train()
    tl = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward(); optimizer.step()
        tl += loss.item()                      
    train_losses.append(tl / len(train_loader))  

    model.eval()
    with torch.no_grad():
        preds = torch.cat([model(xb.to(device)).cpu() for xb, _ in val_loader])
        targs = torch.cat([yb for _, yb in val_loader])
        per_step = ((preds - targs)**2).mean(dim=0)    # (20,) 每步MSE
    print("每步MSE:", per_step.numpy().round(5))

# 画图
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.plot(train_losses, label="train"); plt.plot(val_losses, label="val")
plt.legend(); plt.grid(); plt.xlabel("epoch"); plt.ylabel("MSE")
plt.savefig("loss.png"); print("saved loss.png")

# 基线1: 预测全0 → MSE = 目标的方差
import numpy as np
data = np.load("processed/dataset.npz")
Yva = data["Yva"]
baseline_zero = (Yva**2).mean()                          # 预测0的MSE
baseline_persist = ((Yva[:,1:] - Yva[:,:-1])**2).mean()  # 预测"保持不变"的MSE(粗略)
print("基线-预测0:", baseline_zero, " 基线-持平:", baseline_persist)
print("GRU val_loss:", val_losses[-1])