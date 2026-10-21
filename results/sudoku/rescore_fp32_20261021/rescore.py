"""Re-score saved EMA checkpoints with the repo's own Sudoku eval, decoding in fp32.

Uses eval.sudoku_eval.evaluate_ddp_sudoku and sampling.mdm_sampling unchanged; the
model is wrapped so its forward runs with autocast disabled (fp32 logits, fp32
softmax). Also re-runs top_k_unmasking_1 in bf16 (the original eval path) as a
check against the logged value.
usage (from the repo root): python rescore.py <out.jsonl> <ckpt> [<ckpt> ...]
"""
import sys, json, time, torch
from copy import deepcopy
from omegaconf import OmegaConf

sys.path.insert(0, ".")
from model.transformer import MDMTransformer, MDMConfig
from eval.sudoku_eval import evaluate_ddp_sudoku


class FP32(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        with torch.autocast(device_type="cuda", enabled=False):
            return self.m(x).float()


out_path, ckpts = sys.argv[1], sys.argv[2:]
device = torch.device("cuda")
done = set()
try:
    done = {json.loads(l)["ckpt"] for l in open(out_path)}
except FileNotFoundError:
    pass

for path in ckpts:
    if path in done:
        print("skip", path); continue
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ck["config"])
    model = MDMTransformer(MDMConfig(**ck["config"]["model"])).to(device)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    base = cfg.validation.sampling
    t0 = time.time()
    rec = {"ckpt": path, "step": ck["global_step"], "is_ema": ck.get("is_ema_snapshot"), "fp32": {}}
    for conf in base.confidence:
        for k in base.unmasking_num:
            s = deepcopy(base); s.confidence = conf; s.unmasking_num = k
            rec["fp32"][f"{conf}_unmasking_{k}"] = evaluate_ddp_sudoku(FP32(model), cfg, device, 0, 1, s)
    s = deepcopy(base); s.confidence = "top_k"; s.unmasking_num = 1
    rec["bf16_top_k_unmasking_1"] = evaluate_ddp_sudoku(model, cfg, device, 0, 1, s)
    rec["secs"] = round(time.time() - t0, 1)
    print(json.dumps(rec), flush=True)
    with open(out_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
