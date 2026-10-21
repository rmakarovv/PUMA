"""Decode diagnostics for Sudoku checkpoints (CPU, fp32).

For each EMA checkpoint: greedy top_k decoding with 1 cell/step (same rule as
sampling.mdm_sampling), on the first N test puzzles. Reports exact-grid accuracy
(validity, as in eval), per-cell accuracy vs the stored solution, the #wrong-cells
histogram, when the first error happens, and one-shot per-cell accuracy at fixed
mask fractions (no decoding, like the validation loss).
usage: python decode_diag.py <repo> <N> <ckpt> [<ckpt> ...]
"""
import sys, os, numpy as np, torch, torch.nn.functional as F

repo, N, ckpts = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
sys.path.insert(0, repo)
from model.transformer import MDMTransformer, MDMConfig
from eval.sudoku_eval import verify_sudoku

torch.set_num_threads(int(os.environ.get("NT", 8)))
raw = np.load(os.path.join(repo, "data/sudoku_new/test_mdm.npy"))[:N]
puzzle = torch.from_numpy(raw[:, :81].astype(np.int64))
sol = torch.from_numpy(raw[:, 81:].astype(np.int64))
MASK = 10
BF16 = os.environ.get("BF16") == "1"
empty = puzzle == 0

for path in ckpts:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = MDMTransformer(MDMConfig(**ck["config"]["model"]))
    model.load_state_dict(ck["model_state_dict"]); model.eval()

    with torch.no_grad():
        # greedy decode
        xt = torch.where(empty, torch.full_like(puzzle, MASK), puzzle)
        nsteps = int(empty.sum(1).max())
        first_err = torch.full((N,), -1)
        step_err = torch.zeros(nsteps)  # errors at decode step i (vs solution)
        step_cnt = torch.zeros(nsteps)
        bad_tok = 0; ties = 0
        for i in range(nsteps):
            m = xt == MASK
            if not m.any(): break
            if BF16:
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    logits = model(xt)
                p = F.softmax(logits, dim=-1)  # bf16, as in sampling.mdm_sampling
            else:
                p = F.softmax(model(xt).float(), -1)
            conf, arg = p.max(-1)
            score = torch.where(m, conf, -float("inf"))
            pos = score.argmax(1)
            ties += int(((score == score.max(1, keepdim=True).values) & m).sum(1).gt(1).sum())
            active = m.any(1)
            rows = torch.arange(N)[active]; pos = pos[active]
            tok = arg[rows, pos]
            bad_tok += int(((tok == 0) | (tok == MASK)).sum())
            xt[rows, pos] = tok
            wrong = tok != sol[rows, pos]
            step_err[i] += wrong.sum(); step_cnt[i] += len(rows)
            newly = wrong & (first_err[rows] < 0)
            first_err[rows[newly]] = i
        valid = verify_sudoku(xt, puzzle)
        cell_wrong = ((xt != sol) & empty).sum(1)
        cell_acc = 1 - cell_wrong.sum().item() / empty.sum().item()

        # one-shot per-cell accuracy at fixed mask fractions of the empty cells
        oneshot = {}
        g = torch.Generator().manual_seed(0)
        for frac in (1.0, 0.75, 0.5, 0.25):
            r = torch.rand(N, 81, generator=g)
            r = torch.where(empty, r, 2.0)
            k = (empty.sum(1).float() * frac).round().long().clamp_min(1)
            rank = r.argsort(1).argsort(1)
            mk = rank < k[:, None]
            x = torch.where(mk, torch.full_like(puzzle, MASK), sol)
            pred = model(x).float().argmax(-1)
            oneshot[frac] = ((pred == sol) & mk).sum().item() / mk.sum().item()

    hist = np.bincount(cell_wrong.clamp(max=10).numpy(), minlength=11)
    fe = first_err[first_err >= 0].float()
    print(f"\n{path}  step={ck['global_step']}")
    print(f"  exact(valid)={valid.float().mean():.3f}  exact(=sol)={(cell_wrong==0).float().mean():.3f}  cell_acc={cell_acc:.4f}  bad_tokens={bad_tok} tied_steps={ties}")
    print(f"  #wrong cells hist [0..9,10+]: {hist.tolist()}")
    if len(fe): print(f"  first error step: median={fe.median():.0f} q25={fe.quantile(.25):.0f} q75={fe.quantile(.75):.0f}")
    se = (step_err / step_cnt.clamp_min(1))
    print("  err rate by decode step (bins of 10): " + " ".join(f"{se[j:j+10].mean():.3f}" for j in range(0, nsteps, 10)))
    print("  one-shot cell acc by mask frac: " + " ".join(f"{f}:{a:.4f}" for f, a in oneshot.items()))
