def position_features(deals) -> deal token ids
def action_features(history) -> action token ids
def representation(deal token ids) -> latents
def dynamics(latents, action token ids) -> latents
def prediction(latents) -> prediction token id logits
def decode(prediction token id logits) -> policy, value, next_to_act)

