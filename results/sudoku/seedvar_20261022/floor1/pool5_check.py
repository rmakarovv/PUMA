import sys, collections, torch
sys.path.insert(0, "/home/timur.aisin/intro_to_ml/PUMA-team-BD")
from progressive import PhasedMasking
torch.manual_seed(0)
class Loader:
    def __iter__(self):
        while True:
            pm = torch.zeros(16, 81, dtype=torch.bool)
            for r in pm: r[torch.randperm(81)[:torch.randint(20, 30, (1,)).item()]] = True
            yield {"labels": torch.randint(1, 10, (16, 81)), "prompt_mask": pm}
for kw in [dict(reuse_examples=False), dict(reveal_score="random"), dict()]:
    pool = PhasedMasking(Loader(), 32, 10, 20, torch.device("cpu"), 81, reference_length=81, prompt_only_reset=True,
                         mask_schedule=[81, 61, 41, 21, 1], random_tie_break=True, **kw)
    cnt = collections.Counter(); mins = []
    for t in range(40):
        xt = pool.xt; m = ((xt == 10) & ~pool.state['prompt_mask']).sum(1)
        frac = (m.float() / pool.state['L_eff'].float())
        for s, f in zip(pool.state['stage'].tolist(), frac.tolist()): cnt[(s, round(f, 2))] += 0
        mins.append(m.min().item())
        if t > 5:
            for s, k in zip(pool.state['stage'].tolist(), m.tolist()): cnt[s, k] += 1
        pool.update_with_logits(torch.log_softmax(torch.randn(32, 81, 11), -1))
    by = collections.defaultdict(set)
    for (s, k), v in cnt.items():
        if v: by[s].add(k)
    print(kw or "E", {s: sorted(v) for s, v in sorted(by.items())}, "min masked", min(mins))
