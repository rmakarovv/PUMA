"""GPU check of the candidate eval fix, through the repo's unchanged eval code.

Per checkpoint, top_k_unmasking_1 on the 999 eval puzzles with:
  bf16      model as-is (bf16 autocast forward, bf16 softmax: the current eval)
  fix       bf16 autocast forward, logits cast to fp32 -> fp32 softmax/ranking
  fp32      forward with autocast disabled
Also prints how CUDA topk breaks exact ties.
usage (from the repo root): python fixcheck.py <ckpt> [<ckpt> ...]
"""
import sys, json, torch
from copy import deepcopy
from omegaconf import OmegaConf

sys.path.insert(0, ".")
from model.transformer import MDMTransformer, MDMConfig
from eval.sudoku_eval import evaluate_ddp_sudoku


class Fix(torch.nn.Module):  # forward stays under the caller's bf16 autocast
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        return self.m(x).float()


class FP32(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        with torch.autocast(device_type="cuda", enabled=False):
            return self.m(x).float()


device = torch.device("cuda")
s = torch.ones(81, device=device); s[:5] = 0.3; s[40] = 1.0
print("cuda topk tie ->", torch.topk(s, 1).indices.tolist(), "(lowest tied index is 5)")
s16 = torch.tensor([0.9990, 0.9995, 0.9999], device=device).to(torch.bfloat16)
print("bf16 of [0.9990, 0.9995, 0.9999] ->", s16.float().tolist())

for path in sys.argv[1:]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ck["config"])
    model = MDMTransformer(MDMConfig(**ck["config"]["model"])).to(device)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    smp = deepcopy(cfg.validation.sampling); smp.confidence = "top_k"; smp.unmasking_num = 1
    rec = {"ckpt": path, "step": ck["global_step"]}
    for name, m in (("bf16", model), ("fix", Fix(model)), ("fp32", FP32(model))):
        rec[name] = round(evaluate_ddp_sudoku(m, cfg, device, 0, 1, smp), 4)
    print(json.dumps(rec), flush=True)
