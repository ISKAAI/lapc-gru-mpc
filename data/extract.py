import numpy as np
from asammdf import MDF
from channels import CHANNELS, DT


def read_signals(path):
    m = MDF(path)
    sig = {}
    for name, chan in CHANNELS.items():
        sig[name] = np.asarray(m.get(chan,group = 1).samples, dtype = float)
    return sig

def split_activate_segments(active):
    a = (active != 0).astype(int)
    edges = np.diff(np.concatenate([[0],a,[0]]))
    starts = np.where(edges ==1)[0]
    ends = np.where(edges == -1)[0]
    return list(zip(starts,ends))

def valid(seg):
    for v in seg.values():
        if ( np.isnan(v).any() ):
            return False
    if seg["vx"].min() < 2.0: 
        return False
    if np.abs(seg["c0"]).max() > 2.0:
        return False
    return True

def extract_file(path, trim=20, min_len=100, settle_thr=0.5):
    sig = read_signals(path)
    active = (sig["lcc_active"] !=0) & (sig["alc_active"] == 0)
    segs = []
    for s, e in split_activate_segments(active):
        seg = {k: v[s:e] for k, v in sig.items()}       
        seg = {k: v[:-trim] for k, v in seg.items()}    
        idx = np.where(np.abs(seg["c0"]) < settle_thr)[0]
        jump = np.where(np.abs(np.diff(seg["c0"])) > 0.15)[0]
        if len(jump) > 0:
            seg = {k:v[:jump[0]+1] for k, v in seg.items()} 
        if len(idx) == 0:
            continue
        seg = {k: v[idx[0]:] for k, v in seg.items()}   
        if len(seg["c0"]) < min_len:
            continue
        if not valid(seg):
            continue
        segs.append(seg)
    return segs

if __name__ == "__main__":
    import glob
    for f in sorted(glob.glob("/Users/kai/project/train/alc/*.MF4")):
        segs = extract_file(f)
        print(f"{f.split('/')[-1][:24]} -> {len(segs)}段, 样本{sum(len(s['c0']) for s in segs)}")