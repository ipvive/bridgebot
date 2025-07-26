table view
```mermaid
graph LR;
rap[root action path] --> representation --> dynamics --> prediction --> plwl[possible layouts with likelihood] & agreements & nta[next to act];
fa[future actions] --> dynamics;
```
double-dummy
```mermaid
graph LR;
tg[trick goal] & ac[all cards] & ap[action path] & pa[possible actions] --> network --> value & pal[possible action likelhoods];
pa -->|residual| pal
```
single dummy
```mermaid
graph LR;
TODO;
```
---
* What exactly do the pre- and post- processors do?
  * inputs:
    * par outcome guess, e.g.,
      `[["1", "notrump", "redoubled", "east", "-1"]]`
    * sequence of cards + actions, e.g.,
    ```
        [["West","was dealt", "Club", "Ace"],
        ...
        ["West", "plays", "Club", "Ace"]
        ...
    ]
    ```
  * preprocessor does embedding lookup on each uToken, adds them.
  * outputs:
    * policy
      * policy starts by querying each of the posible actions
      * cross-attention from perceiver latents
    * value >=
    * value >
    * revised par outcome.
  * how to get outputs:
    * for outcome:
      * use `chords.log_likelihood`
    * for `value_geq`, `value_gt`
      * use `chords.log_likelihood` to select between `[YES]`, `[NO]`
    * for policy:
      * use `chords.log_likelihood` to select between `[YES]`, `[NO]`
      * use `target_mask`, `log_softmax` to restricty to legal actions, normalize.
