import numpy as np, sys, os, glob, warnings
warnings.filterwarnings("ignore")
ROOT='/Users/kai/project/lapc_mpc_gru'
sys.path.insert(0,ROOT); sys.path.insert(0,ROOT+'/data'); sys.path.insert(0,ROOT+'/model')
import dataset as DS
from config import VehicleConfig
from model.bicycle_model import BicycleModel

W,Np=20,5
vc=VehicleConfig(); bmodel=BicycleModel(VehicleConfig(Ts=0.05),10.0)
segsXY=[]
for cat in ['lcc','alc']:
    for p in sorted(glob.glob(f'/Users/kai/project/train/{cat}/*.MF4')):
        for seg in DS.extract.extract_file(p):
            X,Y=DS.segment_to_windows(seg,vc,bmodel,W,Np)
            if len(X)>0: segsXY.append((X,Y))
print(f"{len(segsXY)} 段")

def r2_per_step(pred,Y):
    ss=( (pred-Y)**2).sum(0); tot=((Y-Y.mean(0))**2).sum(0)
    return 1-ss/tot

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import HistGradientBoostingRegressor
import torch, torch.nn as nn
from model.gru import DisturbanceGRU

def train_torch(net,Xtr,Ytr,Xva,epochs,seq):
    Xtr_t=torch.tensor(Xtr).float(); Ytr_t=torch.tensor(Ytr).float(); Xva_t=torch.tensor(Xva).float()
    opt=torch.optim.Adam(net.parameters(),lr=1e-3); crit=nn.MSELoss()
    n=len(Xtr_t); idx=np.arange(n)
    for ep in range(epochs):
        np.random.shuffle(idx)
        for i in range(0,n,64):
            b=idx[i:i+64]; opt.zero_grad()
            loss=crit(net(Xtr_t[b]),Ytr_t[b]); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad(): return net(Xva_t).numpy()

class MLP(nn.Module):
    def __init__(s,din=120,h=64,o=5):
        super().__init__(); s.net=nn.Sequential(nn.Linear(din,h),nn.ReLU(),nn.Linear(h,o))
    def forward(s,x): return s.net(x.reshape(len(x),-1) if x.dim()==3 else x)

K=5; rng=np.random.RandomState(0); order=rng.permutation(len(segsXY))
folds=np.array_split(order,K)
models=['Persist','AR(d_last)','Linear','Ridge','Lasso','HistGBR','MLP','GRU']
acc={m:[] for m in models}
for f in range(K):
    va=set(folds[f].tolist()); tr=[i for i in range(len(segsXY)) if i not in va]
    Xtr=np.concatenate([segsXY[i][0] for i in tr]); Ytr=np.concatenate([segsXY[i][1] for i in tr])
    Xva=np.concatenate([segsXY[i][0] for i in folds[f]]); Yva=np.concatenate([segsXY[i][1] for i in folds[f]])
    mean=Xtr.reshape(-1,6).mean(0); std=Xtr.reshape(-1,6).std(0); std[std<1e-8]=1
    Xtr=(Xtr-mean)/std; Xva=(Xva-mean)/std
    Ftr=Xtr.reshape(len(Xtr),-1); Fva=Xva.reshape(len(Xva),-1)
    dtr=Xtr[:,-1,0:1]; dva=Xva[:,-1,0:1]                     # d_last(归一化) 单特征
    acc['Persist'].append(r2_per_step(np.zeros_like(Yva),Yva))
    acc['AR(d_last)'].append(r2_per_step(LinearRegression().fit(dtr,Ytr).predict(dva),Yva))
    acc['Linear'].append(r2_per_step(LinearRegression().fit(Ftr,Ytr).predict(Fva),Yva))
    acc['Ridge'].append(r2_per_step(Ridge(alpha=10).fit(Ftr,Ytr).predict(Fva),Yva))
    acc['Lasso'].append(r2_per_step(Lasso(alpha=1e-4).fit(Ftr,Ytr).predict(Fva),Yva))
    gb=np.column_stack([HistGradientBoostingRegressor(max_iter=150).fit(Ftr,Ytr[:,j]).predict(Fva) for j in range(Np)])
    acc['HistGBR'].append(r2_per_step(gb,Yva))
    torch.manual_seed(0); acc['MLP'].append(r2_per_step(train_torch(MLP(),Xtr,Ytr,Xva,120,False),Yva))
    torch.manual_seed(0); acc['GRU'].append(r2_per_step(train_torch(DisturbanceGRU(),Xtr,Ytr,Xva,60,True),Yva))
    print(f"fold{f} done")

print("\n每步 R² (均值, 越高越好; Persist=0 是基线):")
print("model           step1   step2   step3   step4   step5   | 平均")
for m in models:
    a=np.array(acc[m]); mu=a.mean(0)
    print(f"{m:14s}"+" ".join(f"{v:+.3f}" for v in mu)+f"  | {mu.mean():+.3f}")
print("\n第1步 R² 的 fold间标准差(稳定性):")
for m in models:
    print(f"  {m:14s}{np.array(acc[m])[:,0].std():.3f}")
