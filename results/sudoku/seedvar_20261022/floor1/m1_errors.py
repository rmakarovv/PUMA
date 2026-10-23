import sys, numpy as np, torch, torch.nn.functional as F
repo="/home/timur.aisin/intro_to_ml/PUMA-team-BD"; sys.path.insert(0,repo)
from model.transformer import MDMTransformer, MDMConfig
torch.set_num_threads(16)
raw=np.load(repo+"/data/sudoku_new/test_mdm.npy")[:999]
puz=torch.tensor(raw[:,:81].astype(np.int64)); sol=torch.tensor(raw[:,81:].astype(np.int64))
print("vocab check: puzzle uniq", puz.unique().tolist()[:12], "sol uniq", sol.unique().tolist())
empty=puz==0; N=len(puz); MASK=10
rows=torch.arange(81)//9; cols=torch.arange(81)%9; box=(rows//3)*3+cols//3
peer=((rows[:,None]==rows[None])|(cols[:,None]==cols[None])|(box[:,None]==box[None]))&~torch.eye(81,dtype=bool)
def load(p):
    ck=torch.load(p,map_location="cpu",weights_only=False); m=MDMTransformer(MDMConfig(**ck["config"]["model"])); m.load_state_dict(ck["model_state_dict"]); return m.eval()
@torch.no_grad()
def run(path):
    m=load(path); g=torch.Generator().manual_seed(0)
    # every empty cell of first 200 puzzles, one at a time -> all single masks
    res=[]
    P=200
    for i in range(P):
        idx=empty[i].nonzero().flatten()
        x=sol[i].repeat(len(idx),1); x[torch.arange(len(idx)),idx]=MASK
        lp=F.log_softmax(m(x).float(),-1)
        pred=lp[torch.arange(len(idx)),idx].argmax(-1); conf=lp[torch.arange(len(idx)),idx].max(-1).values.exp()
        for j,c in enumerate(idx.tolist()):
            t=sol[i,c].item(); p=pred[j].item()
            conflict = p in sol[i][peer[c]].tolist()
            res.append((i,c,t,p,conf[j].item(),conflict, int((~empty[i]).sum())))
    r=np.array(res,dtype=float)
    wrong=r[:,2]!=r[:,3]
    print(path.split('runs/')[1], f"single-cell acc {1-wrong.mean():.4f}  n={len(r)}")
    if wrong.any():
        w=r[wrong]
        print("  wrong preds that conflict with a peer:", f"{w[:,5].mean():.3f}", " mean conf wrong", f"{w[:,4].mean():.3f}", "right", f"{r[~wrong][:,4].mean():.4f}")
        print("  pred digit hist (wrong):", np.bincount(w[:,3].astype(int),minlength=11).tolist())
        print("  true digit hist (wrong):", np.bincount(w[:,2].astype(int),minlength=11).tolist())
        print("  err rate by row:", [round(wrong[r[:,1]//9==k].mean(),3) for k in range(9)])
        print("  err rate by col:", [round(wrong[r[:,1]%9==k].mean(),3) for k in range(9)])
        print("  puzzles with any error:", len(set(w[:,0].astype(int))), "of", P)
        pe=[wrong[r[:,0]==i].mean() for i in range(P)]; print("  top puzzle err rates", sorted(pe)[-8:])
for p in sys.argv[1:]: run(p)
