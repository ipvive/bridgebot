import pprint
import bridge.tokens
from mubert import toptokens
from jargon.bert import tokenization as jargon_tokenization

def tokens_to_text(tokens):
  def is_subtoken(word):
    return True if word[:2] == "##" else False
  restored_text = []
  for i in range(len(tokens)):
    if (not is_subtoken(tokens[i]) and
       (i+1) < len(tokens) and is_subtoken(tokens[i+1])):
      restored_text.append(tokens[i] + tokens[i+1][2:])
      if (i+2) < len(tokens) and is_subtoken(tokens[i+2]):
        restored_text[-1] = restored_text[-1] + tokens[i+2][2:]
    elif not is_subtoken(tokens[i]):
      restored_text.append(tokens[i])
  return ' '.join(restored_text)

def features_to_text(d, jargon_vocab_file, jargon_do_lower_case):
  """Converts a features dict to human readable string"""
  player_tokenizer = bridge.tokens.Tokenizer()
  jargon_tokenizer = jargon_tokenization.FullTokenizer(
    vocab_file=jargon_vocab_file,
    do_lower_case=jargon_do_lower_case)
  top_tokenizer = toptokens.Tokenizer()

  def kill_pad(ids):
    return [id for id in ids if id != 0]

  result = [] 
  batch_size = d["player_input_ids"].shape[0] 
  for i in range(batch_size):
    data = {}
    data["player_tokens"] = player_tokenizer.ids_to_tokens(
      kill_pad(d["player_input_ids"][i,:]))
    data["jargon_tokens"] = tokens_to_text(
      jargon_tokenizer.convert_ids_to_tokens(
      kill_pad(d["jargon_input_ids"][i,:])))
    data["query_tokens"] = top_tokenizer.ids_to_tokens(
      kill_pad(d["query_ids"][i,:]))
    data["target_tokens"] = top_tokenizer.ids_to_tokens(
      kill_pad(d["target_ids"][i,:]))
    target_len = len(data["target_tokens"])
    data["target_positions"] = d["target_positions"][i,:target_len]
    data["target_weights"] = d["target_weights"][i,:target_len]

    result.extend("{}: {}".format(k, v) for k, v in data.items())

  return "\n".join(result)
