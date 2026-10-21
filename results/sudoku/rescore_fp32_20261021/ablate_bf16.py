"""Ablate where bf16 decoding hurts (CPU). Greedy top_k, 1 cell/step, like sampling.mdm_sampling.

Variants (forward precision / what the ranking and digit choice use):
  fp32        fp32 forward, fp32 softmax                       (reference)
  rank16      fp32 forward, ranking on bf16-rounded probs, first-index ties (argmax)
  rank16_rnd  fp32 forward, ranking on bf16-rounded probs, random tie-break
  fwd16       bf16 autocast forward, fp32 softmax of the bf16 logits
  bf16        bf16 autocast forward, bf16 softmax (= the repo eval path)
usage: python ablate_bf16.py <repo> <N> <variants,comma> <ckpt> [<ckpt> ...]
"""
import sys, os, numpy as np, torch, torch.nn.functional as F

repo, N, variants, ckpts = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(","), sys.argv[4:]
sys.path.insert(0, repo)
from model.transformer import MDMTransformer, MDMConfig

torch.set_num_threads(int(os.environ.get("NT", 8)))
raw = np.load(os.path.join(repo, "data/sudoku_new/test_mdm.npy"))[:N]
puzzle = torch.from_numpy(raw[:, :81].astype(np.int64))
sol = torch.from_numpy(raw[:, 81:].astype(np.int64))
MASK = 10
empty = puzzle == 0


def step_probs(model, xt, v):
    if v in ("fwd16", "bf16"):
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits = model(xt)
        return F.softmax(logits, -1) if v == "bf16" else F.softmax(logits.float(), -1)
    return F.softmax(model(xt).float(), -1)


def decode(model, v, g):
    xt = torch.where(empty, torch.full_like(puzzle, MASK), puzzle)
    ties = steps = 0
    for _ in range(int(empty.sum(1).max())):
        m = xt == MASK
        if not m.any(): break
        p = step_probs(model, xt, v)
        conf, arg = p.max(-1)
        if v.startswith("rank16"):
            conf = conf.to(torch.bfloat16)
        score = torch.where(m, conf.float(), torch.tensor(-float("inf")))
        top = score.max(1, keepdim=True).values
        tied = (score == top) & m
        if v == "rank16_rnd":
            pos = torch.where(tied, torch.rand(score.shape, generator=g), -1.0).argmax(1)
        else:
            pos = score.argmax(1)
        active = m.any(1)
        ties += int((tied.sum(1) > 1)[active].sum()); steps += int(active.sum())
        rows = torch.arange(N)[active]
        xt[rows, pos[active]] = arg[rows, pos[active]]
    wrong = ((xt != sol) & empty).sum(1)
    return (wrong == 0).float().mean().item(), ties / max(steps, 1)


for path in ckpts:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = MDMTransformer(MDMConfig(**ck["config"]["model"]))
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    with torch.no_grad():
        for v in variants:
            acc, tie_frac = decode(model, v, torch.Generator().manual_seed(0))
            print(f"{path.split('runs/')[-1]}  {v:10s} exact={acc:.3f} tied_step_frac={tie_frac:.2f}", flush=True)
