"""Why does first-index tie-breaking (bf16-rounded ranking) hurt? Trace decodes (fp32 forward, CPU).

Orders: fp32 (true confidence), first (bf16-rounded, lowest index wins), last (bf16-rounded,
highest index wins), rnd (bf16-rounded, random tie-break).
Per order: exact acc; at the first wrong reveal: decode step, #cells still masked, the
chosen cell's fp32 confidence, and the spread of the masked cells over rows at that point
(max masked cells in one row, vs a random mask with the same count).
usage: python trace_order.py <repo> <N> <ckpt> [<ckpt> ...]
"""
import sys, os, numpy as np, torch, torch.nn.functional as F

repo, N, ckpts = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
sys.path.insert(0, repo)
from model.transformer import MDMTransformer, MDMConfig

torch.set_num_threads(int(os.environ.get("NT", 8)))
raw = np.load(os.path.join(repo, "data/sudoku_new/test_mdm.npy"))[:N]
puzzle = torch.from_numpy(raw[:, :81].astype(np.int64))
sol = torch.from_numpy(raw[:, 81:].astype(np.int64))
MASK = 10
empty = puzzle == 0
rows_of = torch.arange(81) // 9


def max_row_load(m):  # [B, 81] bool -> [B] max masked cells in a single row
    return torch.stack([(m & (rows_of == r)).sum(1) for r in range(9)], 1).max(1).values


def decode(model, order, g):
    xt = torch.where(empty, torch.full_like(puzzle, MASK), puzzle)
    fe = dict(step=torch.full((N,), -1), left=torch.zeros(N), conf=torch.zeros(N), rowload=torch.zeros(N), rand_rowload=torch.zeros(N))
    for i in range(int(empty.sum(1).max())):
        m = xt == MASK
        if not m.any(): break
        p = F.softmax(model(xt).float(), -1)
        conf, arg = p.max(-1)
        rank = conf if order == "fp32" else conf.to(torch.bfloat16).float()
        score = torch.where(m, rank, torch.tensor(-float("inf")))
        tied = (score == score.max(1, keepdim=True).values) & m
        if order in ("fp32", "first"):
            pos = score.argmax(1)
        elif order == "last":
            pos = 80 - score.flip(1).argmax(1)
        else:
            pos = torch.where(tied, torch.rand(score.shape, generator=g), -1.0).argmax(1)
        active = m.any(1)
        r = torch.arange(N)[active]; pz = pos[active]
        tok = arg[r, pz]
        wrong = tok != sol[r, pz]
        new = wrong & (fe["step"][r] < 0)
        if new.any():
            rr = r[new]
            fe["step"][rr] = i
            fe["left"][rr] = m[rr].sum(1).float()
            fe["conf"][rr] = conf[rr, pz[new]]
            fe["rowload"][rr] = max_row_load(m[rr]).float()
            # random mask with the same count over the same empty cells
            rs = torch.where(empty[rr], torch.rand(len(rr), 81, generator=g), 2.0)
            k = m[rr].sum(1)
            rm = rs.argsort(1).argsort(1) < k[:, None]
            fe["rand_rowload"][rr] = max_row_load(rm).float()
        xt[r, pz] = tok
    acc = (((xt != sol) & empty).sum(1) == 0).float().mean().item()
    return acc, fe


for path in ckpts:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = MDMTransformer(MDMConfig(**ck["config"]["model"]))
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    with torch.no_grad():
        for order in ("fp32", "first", "last", "rnd"):
            acc, fe = decode(model, order, torch.Generator().manual_seed(0))
            e = fe["step"] >= 0
            print(f"{path.split('runs/')[-1]:50s} {order:5s} exact={acc:.3f} n_err={int(e.sum())}"
                  + (f" | first err: left={fe['left'][e].median():.0f} conf_fp32_median={fe['conf'][e].median():.4f}"
                     f" frac_conf>0.998={(fe['conf'][e] > 0.998).float().mean():.2f}"
                     f" max_row_masked={fe['rowload'][e].mean():.2f} (random mask {fe['rand_rowload'][e].mean():.2f})" if e.any() else ""),
                  flush=True)
