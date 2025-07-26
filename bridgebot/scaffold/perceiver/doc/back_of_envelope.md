conceptually, all answers can be interpreted as marginal
grids of a high dimension tensor.

e.g. [levels.tokens][strains.tokens][seats.tokens][range(4)][(True,False)]

by 
```
possible_vulnerabilities = [
	# South,West,North,East
	[0,0,0,0],
	[0,1,0,1],
	[1,0,1,0],
	[1,1,1,1],
]
```
multimodal_input_encoder will add a separate trainable position encoding per question.
------------
```
inputs = (
  (1,0,1),
  (0,1,1),
)

linearized_inputs = (1,0,1,0,1,1)
pos = ((0,0),(0,1),(0,2),(1,0),(1,1),(1,2))
fourier_pos = (
  (sin(k1*0), ..., cos(kn*0), sin(j1*0), ..., cos(jn*0)),
  (sin(k1*0), ..., cos(kn*0), sin(j1*1), ..., cos(jn*1)),
  (sin(k1*0), ..., cos(kn*0), sin(j1*2), ..., cos(jn*2)),
  (sin(k1*1), ..., cos(kn*1), sin(j1*0), ..., cos(jn*0)),
  (sin(k1*1), ..., cos(kn*1), sin(j1*1), ..., cos(jn*1)),
  (sin(k1*1), ..., cos(kn*1), sin(j1*2), ..., cos(jn*2)),
)
```
--------
```
for vulnerability, want:
 inputs = (1,0,1,0)
fourier_pos = (
   (0, ..., 0, 0, ..., 0, sin(k1*0), ..., cos(k1*0)),
   (0, ..., 0, 0, ..., 0, sin(k1*1), ..., cos(k1*1)),
   (0, ..., 0, 0, ..., 0, sin(k1*2), ..., cos(k1*2)),
   (0, ..., 0, 0, ..., 0, sin(k1*3), ..., cos(k1*3)),
````
=======
```
inputs = (last_bid_level=1, last_bid_strain=notrump, last_bid_seat=west, ...)
encoded = ((0, 1, 0...), (0, ..., 0, 1), ...)
embedded_with_pos = (
   ([num_bands*6], 0),
   ([num_bands*6], 1),
....
)

TOTAL_EMBEDDED_INPUT_SIZE = sum(len(answer_shape))
           * (pixel_embedding_size
              + trainable_question_embedding_size
              + ((2 * num_bands) * (num_labels_sets))
``
to embed "n-s vul, east to act", "east bids 3 clubs":
 * inputs
   * n-s vul data == [1 0 1 0]
   * east to act data == 3
   * action_index == 10
     * level == "3" == 2
     * strain == "Clubs" == 0
 * embedding
             # lvl   strn  seat  pos     Q#      datum
   * n-s vul: [0..0  0..0  fc(0) 0..0] + Q#[0] + emb[1]
   * n-s vul: [0..0  0..0  fc(1) 0..0] + Q#[0] + emb[0]
   * n-s vul: [0..0  0..0  fc(2) 0..0] + Q#[0] + emb[1]
   * n-s vul: [0..0  0..0  fc(3) 0..0] + Q#[0] + emb[0]
   * east:    [0..0  0..0  fc(3) 0..0] + Q#[1] + emb[1]
   * 3 clubs: [fc(2) fc(0) fc(3) 0..0] + Q#[2] + emb[1]
---
use cases for dims:
 * absolute
   * abridged-bidding
     * rebuild FSA
     * use FSA to forward answers to target state
     * use FSA to train perceiver
   * full-bidding
     * train perceiver directly from game.
   * bidding-and-play
     * train perceiver directly from game.
 * absolute + fuzzy
   * ...
   

period of 5

dim val 

    0 1 2 3 4

    frequencies
     
    f0 f1          f4   
    0  2pi/5 ... 4 * 2pi/5
    0  2pi/10 ...... 9 * 2pi / 10
    etc

    (0 1/5 ..  9/5) pi

    (1 6/5 ..  14/5) pi

    embedding

    sin(fi * dim) cos(fi * dim) for not None

Action info in the model
* Use first three question as action information
    * two of them will be none for each training example
* What tensors should the decoder query include for cross attention?

Model Inputs
 * generate from online_fsa
 * model inputs is `dict(action_answers=tuple(), 
    current_state_answer=tuple(), next_state_answers=tuple())`
    * do we want to use tf.Dataset? YES
    * do we want to keep action_answers and state_answers separate? YES
    * do we want a separate datasource.py? YES
    * do we want separate eval and train datasets (i.e. split)? Mostly NO
        * for effectively finite - NO. Want to see if can learn entire dataset
        * [YAGNI] for effectively infinite - Kinda YES (need generate fixed dataset) 
    
---
loss function design
 * different for one-hot, absolute, fuzzy answers.
 * 4 experiments.py:
   * fsa-bidding (all one-hot)
   * full-bidding (one-hot + 1x absolute)
   * full-game (one-hot + ~7x absolute)
   * game-plus (+ ~5-20x fuzzy)
 * write dims_perceiver.one_hot_loss
 * write dims_perceiver.loss_fn
 * metrics:
   * overall (loss, top_1_acc)
   * (loss, top_1_acc) per question
   * (loss, top_1_acc) per question and dim

future_state:
   dict:
      question_name -> batched_answer_array

logits:
    dict:
      question_name -> dict: dim_name -> batched_logits)

what do we do with the logits?
 * [YES] we might like to expose them for training.
 * [YES] we want discrete choices for inference + eval.
how does postprocess work for last_bid?
 * [NO] naively we could spam position logits
   for each of 4*5*4*3=240 possible positions.
 * [YES] alternatively we could treat each dimension separately,
   using just 4+5+4+3=16 position logits.
when is_training=True and question_names=('last_bid', 'stage'),
    what is the structure of for return value?
 * ```
   {
     'last_bid': {
       'level': np.array(logits, shape=(7,)),
       'strain': np.array(logits, shape=(5,)),
       'seat': np.array(logits, shape=(4,)),
       'call': np.array(logits, shape=(3,)),
     },
     'stage': {
       'stage': np.array(logits, shape=(4,)),
     },
   }
  ```
-------
None/NA/-1 treatment.
 * Game logic *MUST NOT* depend on these values.
   * add variables as needed, e.g., bidding_is_open.
   * ensure they are irrelevant with peturbation testing.
 * loss fn should masks off predictions where ezpexted=-1.
 * we should augment training data with -1->random.
------
we plan to maintain a set of A/B experiments where we
face uncertainty, beginning with FLAGS.experiment_none_is_uniform.
note such sites with a FLAG.experiment_...,
unimplemented unleess time permits.
   
