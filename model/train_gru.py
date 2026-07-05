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
        per_step = ((preds - targs)**2).mean(dim=0)    # (Np,) 每步MSE
    val_losses.append(per_step.mean().item())          # 记录平均每步MSE, 供画图/打印
    last_per_step = per_step.numpy()                    # 留给下面的基线对比
    print("每步MSE(增量):", last_per_step.round(6))

# 画图
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.plot(train_losses, label="train"); plt.plot(val_losses, label="val")
plt.legend(); plt.grid(); plt.xlabel("epoch"); plt.ylabel("MSE")
plt.savefig("loss.png"); print("saved loss.png")

# 基线对比: Yva 现在是"相对窗口末端 d_last 的增量"
#   在增量空间里, 持久化前馈 <=> 预测增量恒为 0, 其每步MSE = 增量的方差.
#   => GRU 只要每步 MSE 低于该基线, 就是真正跑赢了持久化.
Yva = data["Yva"]                              # 增量目标
base_persist = (Yva**2).mean(axis=0)           # 持久化(预测Δ=0) 每步MSE
print("持久化(预测Δ=0) 每步MSE:", base_persist.round(6))
print("GRU(学增量)     每步MSE:", last_per_step.round(6))
improve = 1.0 - last_per_step / base_persist   # >0 表示比持久化好, 数值=相对降幅
print("每步相对持久化降幅:", (improve*100).round(1), "%   (正=赢, 负=没赢)")
print("平均: GRU=%.6f  持久化=%.6f  ->  %s" % (
      last_per_step.mean(), base_persist.mean(),
      "GRU 赢 √" if last_per_step.mean() < base_persist.mean() else "没赢 ×"))