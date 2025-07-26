import glob
import lmppl
import torch
import numpy as np
fns = glob.glob('data/*.lin')
def read_file(fn):
    with open(fn) as f:
        return f.read()
texts = [read_file(fn) for fn in fns]
concat_size = 10
longtexts = []
for s in range(0, len(texts), concat_size):
    longtexts.append("".join(texts[s:s + concat_size]))
scorer = lmppl.LM("Qwen/Qwen2.5-1.5B-Instruct")
with torch.device("cuda:0"):
    ppl = scorer.get_perplexity(texts)
    lppl = scorer.get_perplexity(longtexts)
print(f"perplexity: mean {np.mean(ppl)} +/- {np.std(ppl)}")
print(f"perplexity - concatenated: mean {np.mean(lppl)} +/- {np.std(lppl)}")
