 * inputs:
   * case 1: bbo-hand-records: .lin, DBs
   * case 2: bm.json, DBs
 * step 1: representation. given the history, public cards,
   and private cards (if provided),
   cross-attend + self-attend to get latents.
 * step 2: dynamics. given latents, action observables, get latents, `next_to_act`, `is_irreular`
 * step 3: rag (skip for now). latents --> latents.
 * step 4: cross-attend prefix, latents --> latents.
 * step 5: cross-attend latents --> decoder-seed
 * step 6: generate

