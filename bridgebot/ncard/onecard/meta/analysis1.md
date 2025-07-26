### goal
 * [[[find e-equilibrium perceiver net by 6 jan]]]
### target
 * method: algorithmic supervision for all but N decisions
 * "value"
   * win/lose/tie other table
 * "policy"
   * bid|call|play priority (dims)
 * inputs:
   * game state (dims)
### next steps
 * [@li] [T.nn.1] test+implement AbsoluteObservable
 * [DONE] [T.reason.1] execute seed.kg
 * [AFTER jax.experimental.xmap out of aspirational stage]
   * [T.rdims.1] tdd rgeom as minimally needed for seed.kg execution.
 * [@ay] [T.experiment.1] implement bootstrap learning in jaxline
   * [@nk] [T.experiment.1.1] value,policy perceiver end
### subgoal analysis
