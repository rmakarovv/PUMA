"""GPU check of the sampling.py patch: the original (git HEAD) and the patched mdm_sampling decode the
same 999 eval puzzles through the repo's evaluate_ddp_sudoku (top_k_unmasking_1).
usage (from the patched repo root): python check_patch.py <sampling_orig.py> <ckpt> [<ckpt> ...]"""
import sys, importlib.util, torch
from copy import deepcopy
from omegaconf import OmegaConf
sys.path.insert(0, ".")
from model.transformer import MDMTransformer, MDMConfig
import eval.sudoku_eval as se
import sampling as patched
spec = importlib.util.spec_from_file_location("sampling_orig", sys.argv[1]); orig = importlib.util.module_from_spec(spec); spec.loader.exec_module(orig)
dev = torch.device("cuda")
for p in sys.argv[2:]:
    ck = torch.load(p, map_location="cpu", weights_only=False); cfg = OmegaConf.create(ck["config"])
    m = MDMTransformer(MDMConfig(**ck["config"]["model"])).to(dev); m.load_state_dict(ck["model_state_dict"]); m.eval()
    s = deepcopy(cfg.validation.sampling); s.confidence = "top_k"; s.unmasking_num = 1
    out = {}
    for name, mod in (("original", orig), ("patched", patched)):
        se.mdm_sampling = mod.mdm_sampling  # evaluate_ddp_sudoku calls the module-level name
        out[name] = round(se.evaluate_ddp_sudoku(m, cfg, dev, 0, 1, s), 4)
    print(p.split("runs/")[-1], out, flush=True)
