 * simulation
     * ```python
# MCCFR
def simulate(history, info):
  repeat N times:
    if need less infogame uncertainty:
      sample and weight new related position consistent with public state.
        ?exclude analyzed partitions?
    use policy to choose paths in game forest; estimate likelihood + relevancy.
      if need policy:
        estimate using augmented neural net
      when close enough to end and still have sufficient samples:
        use exhaustive solver to calculate value, partition
      otherwise, if tree is too sparse:
        use augmented neural net to estimate value, partition
      if have value;
        backpropogate value & partition along (reversed) chosen path(s)
          compute regret at each node
          apply sampled CFR to update node's policy
```
 * runtime
   * ```python
def play():
  simulate()
  choose action based on updated policy at consistent samples
```
 * learning
   * ```
def train():
  # roll out games until exhaustively solvable.
  # predict updated policy, region, value at roots of simulations.
```
 * EV
   * depends on input likelihoods
     * fn is continuous, piecewise linear
     * input likelihoods are estimates, not precisely known.
     * model as affine fn of likelihoods around nbhd of estimate.
   * value :=
     ?EV+gradient of chosen path for actual deal?
     ?EV+gradient of chosen paths for sampled deals?
 * concerns:
   * devil in the details.
   * may require too much compute.
   * need better way to focus on relevant positions
   * too much spaghetti logic
