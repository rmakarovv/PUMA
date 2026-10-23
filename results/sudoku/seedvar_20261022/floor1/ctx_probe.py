# Accuracy on one target empty cell c as a function of (a) total masked count m and (b) where the other m-1 masks sit:
# "far" = empty cells that are not peers of c, "near" = empty peers of c first.
import sys, numpy as np, torch
repo="/home/timur.aisin/intro_to_ml/PUMA-team-BD"; sys.path.insert(0,repo)
from model.transformer import MDMTransformer, MDMConfig
torch.set_num_threads(16)
raw=np.load(repo+"/data/sudoku_new/test_mdm.npy")[:999]
puz=torch.tensor(raw[:,:81].astype(np.int64)); sol=torch.tensor(raw[:,81:].astype(np.int64))
empty=puz==0; N=len(puz); MASK=10
r_=torch.arange(81)//9; c_=torch.arange(81)%9; b_=(r_//3)*3+c_//3
peer=((r_[:,None]==r_[None])|(c_[:,None]==c_[None])|(b_[:,None]==b_[None]))&~torch.eye(81,dtype=bool)
g=torch.Generator().manual_seed(1)
# fixed target and random priorities per puzzle
tgt=torch.stack([e.nonzero().flatten()[torch.randint(int(e.sum()),(1,),generator=g)] for e in empty]).flatten()
pri=torch.rand(N,81,generator=g)
def board(m, where):
    x=sol.clone(); x[torch.arange(N),tgt]=MASK
    for i in range(N):
        c=tgt[i]; cand=empty[i].clone(); cand[c]=False
        if where=="far": cand&=~peer[c]; order=torch.where(cand,pri[i],9.0).argsort()
        else: order=torch.where(cand, pri[i]-2.0*peer[c].float(),9.0).argsort()  # peers first
        k=min(m-1,int(cand.sum())); x[i,order[:k]]=MASK
    return x
def load(p):
    ck=torch.load(p,map_location="cpu",weights_only=False); mdl=MDMTransformer(MDMConfig(**ck["config"]["model"])); mdl.load_state_dict(ck["model_state_dict"]); return mdl.eval()
boards={(m,w):board(m,w) for m in [1,2,4,8,14,20] for w in ["far","near"]}
with torch.no_grad():
    for p in sys.argv[1:]:
        mdl=load(p); out=[]
        for (m,w),x in boards.items():
            pred=mdl(x).float().argmax(-1)[torch.arange(N),tgt]
            out.append(f"{w}{m}:{(pred==sol[torch.arange(N),tgt]).float().mean()*100:.1f}")
        print(p.split("runs/")[1], " ".join(out), flush=True)
