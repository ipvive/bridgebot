        * shape 1st, 4321 points 2nd, 21 points 3rd, aces 4th, cards 5th.
        * group by futures needed in error-free play + 
        * stratified sample, with weight.
        * sample entire batch at once w/o reference to private info.
        * sequence:
          ```
          1. determine shapes
             a. if known.
             b. else ask agents x 4 to infer.
          2(a,b) determine 4321 points values
          3(a,b), 4(a,b), 5(a,b)
          group
          prune (stochastic; upweight if prune candidate)
          validate vs. policy.
          ```
for 1+2card, card classes are not useful.
for 3card, need ability to ciunt #AK., #KQ.
determine card interchange by inference
ML repr == bitmap(N==N-1i // miniax trick counti; or  --> and, exceept break ties in same syut,)
for 1card, brute force sufficieent. YAGNI.
---
@dataclass
class PositionCluster:
    min_length: tuple[int]
    max_shape: tuple[int]
    rank_classes  # bitmap: is card N interchangeable with card n-1?
    ;
    rank_classes
    rank_class_count: map[tuple[int], int]  # for each hand, for each rank, for each suit: class
    example: bridgegame.Deal

class PositionGenerator:
    def get(num_to_return, num_to_sample):


---
def generate_deals(start_deal) -> list[PositionGlob]
def infer(list[PositionGlob]) -> list[
