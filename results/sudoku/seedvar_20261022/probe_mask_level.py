"""Why do some B/D seeds fail? Probe final EMA checkpoints at controlled masking levels (fp32, CPU).

1. Re-mask probe: take the solved grid, mask m of the puzzle's empty cells uniformly at random (same cells for
   every model: fixed generator), one forward, per-cell argmax accuracy on the masked cells. m is either an
   absolute count (1, 2, 3, 5, 8) or a fraction of the empty cells (0.26 = the smallest training state of
   E/D/B, 0.5, 0.75, 1.0).
2. fp32 greedy decode (top_k, 1 cell per step, as the proposal decoder): exact-grid accuracy and, per puzzle,
   the number of masked cells left when the first wrong cell is revealed.
Runs on CUDA when available (fp32, no autocast), else CPU.
usage: python probe_mask_level.py <repo> <N> <out.jsonl> <tag>=<ckpt> [<tag>=<ckpt> ...]
"""
import json, os, sys

import numpy as np
import torch
import torch.nn.functional as F

repo, N, out_path, items = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4:]
sys.path.insert(0, repo)
from model.transformer import MDMTransformer, MDMConfig

torch.set_num_threads(int(os.environ.get("NT", 16)))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = False  # true fp32 matmuls
raw = np.load(os.path.join(repo, "data/sudoku_new/test_mdm.npy"))[:N]
puzzle = torch.from_numpy(raw[:, :81].astype(np.int64)).to(dev)
sol = torch.from_numpy(raw[:, 81:].astype(np.int64)).to(dev)
MASK = 10
empty = puzzle == 0
n_empty = empty.sum(1)
LEVELS = [1, 2, 3, 5, 8, 0.26, 0.5, 0.75, 1.0]


def remask(level, g):
    k = torch.full((N,), level, device=dev) if isinstance(level, int) else (level * n_empty.float()).round().long().clamp_min(1)
    r = torch.where(empty, torch.rand(N, 81, generator=g).to(dev), 2.0)
    m = r.argsort(1).argsort(1) < k[:, None]
    return torch.where(m, torch.full_like(sol, MASK), sol), m


@torch.no_grad()
def forward(model, x, bs=500):
    return torch.cat([model(x[i:i + bs]).float() for i in range(0, len(x), bs)])


@torch.no_grad()
def decode(model):
    xt = torch.where(empty, torch.full_like(puzzle, MASK), puzzle)
    first_left = torch.full((N,), -1, device=dev)
    for _ in range(int(n_empty.max())):
        m = xt == MASK
        if not m.any():
            break
        p = F.softmax(forward(model, xt), -1)
        conf, arg = p.max(-1)
        pos = torch.where(m, conf, -1.0).argmax(1)
        active = m.any(1)
        r = torch.arange(N, device=dev)[active]
        pz = pos[active]
        tok = arg[r, pz]
        new = (tok != sol[r, pz]) & (first_left[r] < 0)
        first_left[r[new]] = m[r[new]].sum(1)
        xt[r, pz] = tok
    return (xt == sol).all(1).float().mean().item(), first_left


for item in items:
    tag, path = item.split("=", 1)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = MDMTransformer(MDMConfig(**ck["config"]["model"]))
    model.load_state_dict(ck["model_state_dict"])
    model.to(dev).eval()
    g = torch.Generator().manual_seed(0)
    rec = {"tag": tag, "device": str(dev), "ckpt": path, "step": ck["global_step"], "cell_acc": {}}
    for level in LEVELS:
        x, m = remask(level, g)
        pred = forward(model, x).argmax(-1)
        rec["cell_acc"][str(level)] = ((pred == sol) & m).sum().item() / m.sum().item()
    acc, first_left = decode(model)
    fl = first_left[first_left >= 0].cpu()
    rec["decode_exact"] = acc
    rec["first_error_left_hist"] = {f"{a}-{b}": int(((fl >= a) & (fl <= b)).sum()) for a, b in
                                    [(1, 1), (2, 3), (4, 8), (9, 15), (16, 30), (31, 81)]}
    print(json.dumps(rec), flush=True)
    with open(out_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
