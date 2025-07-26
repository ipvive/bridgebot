from transformers import RobertaTokenizer, FlaxRobertaForMaskedLM

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

model = FlaxRobertaForMaskedLM.from_pretrained("roberta-base")

inputs = tokenizer("The capital of France is <mask>.", return_tensors="jax")

outputs = model(**inputs)

logits = outputs.logits

vocab = tokenizer.get_vocab()
rvocab = {v:k for k,v in vocab.items()}
answer = [rvocab[int(logits[0,i].argmax().to_py())] for i in range(11)]
print(answer)
