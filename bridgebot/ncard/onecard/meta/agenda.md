### Goals
 * train e-equilibrium perceiver net for 1card.
### tasks
 * [DONE] [rules] test and implement rules.
 * [scaffold] fork & update scaffold.
   * `dims.py`: needed changes only.
   * `dims_perceiver.py`: AbsoluteObservable.
   * `experiment.py`: rewrite.
 * [exploit] code exploitabiility metric.
 * [train] train & eval
```mermaid
train --depends--> rules & exploit & scaffold
```
