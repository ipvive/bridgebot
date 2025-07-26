"""
Task A: compute exact value and policy of a given double-dummy end position.
 * solved ~1998
Task A1. A if feasible within budget, else estimate 
 * needed?
Task B: find a maximal set of "weakly equivalent" deals.
Task C: compute exact value and policy of the related single-dummy positions."
Task C1. C if feasible within budget, else estimate 
"""
# * we will need to supply inference_pipe[2].
# * we will need to fork (or generalize) simulate.py?
# * how quickly can we get to SVD/clustering/memory?
#   * does a different task make more sense?
# * how do we cluster?
#   * moot card permutation equivalence
# * do we want to use high-c

# ---

# * background:
#   * [Gitelman 1996](https://www.aaai.org/Papers/AAAI/1996/AAAI96-034.pdf)
#   * [Beling 2017](http://pbeling.w8.pl/en/publication/2017_partition_search_revisited/)
# basic-partition = orthogonal-tensor-bsp
#           := intersection(+-((sum(contiguous_slice(state_tensor))) + K) >= 0)
#   A- relax {contiguous_,}slice
#   B- Union p_i if all p_i travel in parallel for M moves.
#   C- convex polytope in frequency simplex with partition vertices.
#     (mandatory deception / mixed play sub-optimal in interior)
#   D- point in frequency simplex with mixed play required.

# ---

# trick permutation
#  idea: cluster paths on set of trick-winning card partitions.
#        or set of (starting, winning) pairs of card partitions.
