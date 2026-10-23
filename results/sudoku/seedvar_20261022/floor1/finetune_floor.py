"""Causal check for the B/D low-mask failure: continue training a final EMA checkpoint for N updates with the
condition's own pool (PhasedMasking, B: reuse_examples=False, D: reveal_score=random) under either
  sched4 = (81, 61, 41, 21)      the original schedule (control: extra training only), or
  sched5 = (81, 61, 41, 21, 1)   the schedule plus a 1-mask state,
then measure single-cell accuracy (m = 1, 2, 3, 8 random empty cells re-masked, same cells for every model) and fp32
greedy top_k decode exact-grid accuracy on the 999 test puzzles (as probe_mask_level.py).
Single process, batch 64 (= 2 GPUs x 32 in the runs), AdamW constant lr, fp32. Data: training split of the repo
loader (split_sudoku, seed 2026). usage:
  python finetune_floor.py <repo> <ckpt> <B|D> <sched4|sched5> <steps> <lr> <seed> <out.jsonl> <tag>
"""
import json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F

repo, ckpt, cond, sched, steps, lr, seed, out_path, tag = sys.argv[1:10]
steps, lr, seed = int(steps), float(lr), int(seed)
sys.path.insert(0, repo)
os.chdir(repo)
from model.transformer import MDMTransformer, MDMConfig
from progressive import PhasedMasking, mdm_loss_fn
from data.sudoku import split_sudoku

torch.set_num_threads(int(os.environ.get("NT", 16)))
torch.manual_seed(seed)
MASK = 10
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
model = MDMTransformer(MDMConfig(**ck["config"]["model"]))
model.load_state_dict(ck["model_state_dict"])

schedule = {"sched4": [81, 61, 41, 21], "sched5": [81, 61, 41, 21, 1]}[sched]
kw = {"B": dict(reuse_examples=False), "D": dict(reveal_score="random")}[cond]
train_data, _ = split_sudoku("data/sudoku_new", "new", val_ratio=0.05, seed=2026, mmap=False)
loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True,
                                     generator=torch.Generator().manual_seed(seed))
pool = PhasedMasking(loader, 64, MASK, 20, torch.device("cpu"), 81, reference_length=81, prompt_only_reset=True,
                     mask_schedule=schedule, random_tie_break=True, **kw)
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

t0 = time.time()
model.train()
losses = []
for step in range(steps):
    xt = pool.current_batch()
    log_probs = F.log_softmax(model(xt), dim=-1)
    loss = mdm_loss_fn(log_probs, pool.x0, pool.xt, MASK, prompt_mask=pool.state['prompt_mask'])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    pool.update_with_logits(log_probs.detach())
    losses.append(loss.item())
    if step % 100 == 0:
        print(f"{tag} step {step} loss {np.mean(losses[-100:]):.4f} {time.time() - t0:.0f}s", flush=True)
model.eval()

raw = np.load("data/sudoku_new/test_mdm.npy")[:999]
puzzle = torch.from_numpy(raw[:, :81].astype(np.int64))
sol = torch.from_numpy(raw[:, 81:].astype(np.int64))
empty = puzzle == 0
N = len(puzzle)
n_empty = empty.sum(1)


@torch.no_grad()
def forward(x, bs=500):
    return torch.cat([model(x[i:i + bs]).float() for i in range(0, len(x), bs)])


g = torch.Generator().manual_seed(0)
cell_acc = {}
for m in [1, 2, 3, 8]:
    r = torch.where(empty, torch.rand(N, 81, generator=g), 2.0)
    msk = r.argsort(1).argsort(1) < m
    x = torch.where(msk, torch.full_like(sol, MASK), sol)
    pred = forward(x).argmax(-1)
    cell_acc[str(m)] = ((pred == sol) & msk).sum().item() / msk.sum().item()

with torch.no_grad():
    xt = torch.where(empty, torch.full_like(puzzle, MASK), puzzle)
    first_left = torch.full((N,), -1)
    for _ in range(int(n_empty.max())):
        msk = xt == MASK
        if not msk.any():
            break
        conf, arg = F.softmax(forward(xt), -1).max(-1)
        pos = torch.where(msk, conf, -1.0).argmax(1)
        rows = torch.arange(N)[msk.any(1)]
        pz = pos[rows]
        tok = arg[rows, pz]
        new = (tok != sol[rows, pz]) & (first_left[rows] < 0)
        first_left[rows[new]] = msk[rows[new]].sum(1)
        xt[rows, pz] = tok
exact = (xt == sol).all(1).float().mean().item()
fl = first_left[first_left >= 0]
rec = {"tag": tag, "ckpt": ckpt, "cond": cond, "sched": sched, "steps": steps, "lr": lr, "seed": seed,
       "train_loss_last100": float(np.mean(losses[-100:])) if losses else None, "cell_acc": cell_acc,
       "decode_exact": exact, "first_error_at_1_left": int((fl == 1).sum()), "first_error_2to3": int(((fl >= 2) & (fl <= 3)).sum()),
       "first_error_ge4": int((fl >= 4).sum()), "time_s": time.time() - t0}
print(json.dumps(rec), flush=True)
with open(out_path, "a") as f:
    f.write(json.dumps(rec) + "\n")
