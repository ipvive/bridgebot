import glob
import lmppl
import torch
import numpy as np
fns = glob.glob('data/*.lin')
def read_file(fn):
    with open(fn) as f:
        return f.read()
texts = [read_file(fn) for fn in fns]
scorer = lmppl.LM("Qwen/Qwen2.5-1.5B-Instruct")
with torch.device("cuda:0"):
    ppl = scorer.get_perplexity(texts)
print(f"perplexity: mean {np.mean(ppl)} +/- {np.std(ppl)}")
