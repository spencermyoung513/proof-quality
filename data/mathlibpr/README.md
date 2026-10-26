# MathlibPRProofPairs

This dataset is sourced from the `pairs` split of [MathlibPR/MathlibPR](https://huggingface.co/datasets/MathlibPR/MathlibPR). The split originally contained 3,687 pairs of `Mathlib` pull requests where:

- The first item passed build checks (linting, compilationk, etc.) but was rejected by reviewers
- The second item (a revision of the first) was accepted by reviewers and merged into `Mathlib`

We have converted this dataset into explicit declaration-proof-proof triples. Each item is a JSON object with the following keys:

- `prompt` — `"Write a Mathlib-quality Lean 4 proof of the following declaration."`, plus the statement to prove, ending in `:=`.
- `chosen` — the accepted proof
- `rejected` — the earlier version of the proof that was part of the rejected PR

The triples are stored in `pairs.jsonl`. Our code extracted 2,379 in total. For associated metadata, see `summary.json`.

The dataset was built by Cursor's agent on 10/23/2026. It scanned the 3,687 pairs from MathlibPR, looking for statements (`theorem` or `lemma`) whose declaration remained unchanged but had a different proof at submission vs. acceptance time. The associated code for doing this is in `proof-quality/src/proof_quality/dataset_generation`.

## How we identify valid declaration-proof-proof triples

Our procedure parses the Lean source code from a partial clone of `leanprover-community/mathlib4`.

Each snapshot in `MathlibPR` is one commit of a pull request, with a base commit specified by `diff_base_sha`. For each pair of PRs in the `pairs` split, we compare each snapshot with its base commit, and scan for any new or updated `theorem` or `lemma`. The declaration has to be present in both snapshots. We also record the enclosing `namespace`, `section`, and `mutual` block.

Sometimes, the code contains `variable` commands before a theorem to supply types and hypotheses to the declarations that follow. Lean automatically inserts these binders wherever the theorem references the declared name, but to make it so each example can stand alone in our data without the rest of the file, we replace variable references with their actual values in the statement.

Lean code often repeats certain names to make the code more explicit. For example, in `variable {s: Set α} (s)`, the first group introduces `s` and its type and the second marks that same `s` as an explicit argument. When we encounter this scenario while forming our triples, we copy the first group into the statement and skip the second.

We keep a declaration-proof-proof triple only when all of these hold:

1. The declaration is a `theorem` or a `lemma`.
2. The declaration name is contained exactly once in both snapshots.
3. The rejected snapshot changes lines inside the declaration. Either the accepted snapshot changes it too, or the accepted snapshot leaves it in place and its proof is the proof from the rejected snapshot's base commit. The second case is a proof edit that the revision dropped.
4. The statements for each declaration match (after removing whitespace).
5. After removing whitespace, the text content of each proof differs. This includes any added/updated/deleted comments.
6. Neither proof is `sorry` or `by sorry`.



## Command to reproduce our dataset from scratch

```bash
uv run python -m proof_quality.dataset_generation \
  --pairs <path/to/within_pr_pairs_full.jsonl> \
  --repo <path/to/mathlib4-partial-clone> \
  --output <desired/path/for/pairs.jsonl> \
  --summary <desired/path/for/summary.json> \
  --dataset-version mathlibpr-v1 \
  --fetch
```

`within_pr_pairs_full.jsonl` can be downloaded from [the MathlibPR dataset](https://huggingface.co/datasets/MathlibPR/MathlibPR/blob/main/data/pairs/within_pr_pairs_full.jsonl) on Huggingface.

The `--fetch` argument pulls any missing commits into the repo clone via `git fetch --filter=blob:none`.

## Some facts about the dataset


|                                        |                                    |
| -------------------------------------- | ---------------------------------- |
| PRs scanned                            | 3,687                              |
| PRs with at least one kept declaration | 923                                |
| Kept declarations                      | 2,379 (1,557 theorems, 822 lemmas) |


During the scan, we encountered several declarations that did not make it into our final dataset. Here are the reasons, along with the # of declarations that were excluded:


| Reason              | Count  | Description                                                                                                                   |
| ------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------- |
| `unchanged_proof`   | 33,597 | The declaration is in both snapshots, but the proofs are identical.                                                           |
| `one_sided`         | 5,723  | The declaration is changed in the rejected snapshot, but is not present at all (with the same name) in the accepted snapshot. |
| `signature_changed` | 1,651  | The statements differ after whitespace is removed.                                                                            |
| `not_reverted`      | 588    | The accepted snapshot still has the declaration, but its proof is not the proof from before the rejected edit.                |
| `upstream`          | 44     | Both proofs already match what is on their base commits; any differences come from the base commit moving, not the PR itself. |
| `ambiguous_name`    | 32     | The declaration name occurs more than once in a snapshot, so a unique pair cannot be formed.                                  |
| `sorry`             | 1      | At least one proof is `sorry` or `by sorry`.                                                                                  |




## Filtering convention edits

Many of the triples in `pairs.jsonl` differ because a revision followed a Mathlib convention, not because the two proofs are different arguments. The filter compares each pair after separating comments from code and normalizing those conventions. It keeps a triple when the proof script changes shape, or when the code matches and the comment wording itself changes.

A script changes shape when a real step appears or disappears: `calc`, `induction`, `cases`, `ext`, or a `have` or `let` whose body is itself a proof. A hammer such as `grind` or `omega` swapped for a written proof counts too. `aesop` and `cat_disch` are the same discharger. The same tactics aimed at a different lemma do not count, and neither does a `have` or `let` that only names a term or an instance. Comment edits that only reflow text, including `n + 1` versus `n+1`, are dropped. A convention change in the code is still dropped when a comment changes along with it.

What gets dropped is the convention layer: renames and coercions (`(n : ℕ)` versus `n`, `eq_zero_or_pos` versus `eq_or_ne`), lemmas moved inside one `simp` or `rw` list, tactic synonyms (`apply`/`exact`, `field_simp`/`simp`, `obtain`/`cases`, `induction'` versus structured `induction`), reordered bullets, layout (`@`, pipes, parentheses, `by exact` versus a bare term), and the same proof with new binders, holes, projections, or explicit arguments. Folding a term-mode `have` into `simp`, writing `#s` for `s.card`, or replacing borel boilerplate with `borelize` is the same kind of edit.

`pairs.substantive.jsonl` is the kept subset (539 of 2,379). `pairs.annotated.jsonl` labels every triple with `metadata.diff_class`. Regenerate both with `python -m proof_quality.dataset_generation.filter`.


| Class             | Rows | Share | Verdict | What it is                                                                         |
| ----------------- | ---- | ----- | ------- | ---------------------------------------------------------------------------------- |
| `construction`    | 519  | 21.8% | keep    | The script gains or loses a real step, or a hammer is swapped for a written proof. |
| `comment`         | 20   | 0.8%  | keep    | The Lean code matches, and the comment wording itself changes.                     |
| `comment_spacing` | 15   | 0.6%  | drop    | The code matches, and the comments differ only by spacing or reflow.               |
| `style`           | 113  | 4.7%  | drop    | Tactic synonyms, `@`, pipes, or `by exact` versus a bare term.                     |
| `layout`          | 125  | 5.3%  | drop    | Parentheses, bullets, or other punctuation.                                        |
| `rename`          | 528  | 22.2% | drop    | An identifier or namespace edit, including one lemma name replaced in place.       |
| `simp_list`       | 186  | 7.8%  | drop    | Lemmas added or removed inside one `simp` or `rw` list.                            |
| `bookkeeping`     | 127  | 5.3%  | drop    | Binders, holes, projections, or explicit arguments on the same proof.              |
| `minor`           | 746  | 31.4% | drop    | A non-structural tweak, such as deleting a redundant `rw`.                         |


### On length bias

A natural / "dumb" heuristic for a proof that gets accepted into Mathlib is "the shorter, the better". Indeed, this is often the case when a more elegant set of tactics may achieve the same goal. However, we also want to capture the cases where this rule doesn't hold, since there are other quality signals in a proof besides length. In the filtered `pairs.substantive.jsonl`, 430 triplets (79.8%) have a chosen proof shorter than the rejected one, 107 (19.9%) have a chosen proof that is longer, and 2 are the same length.

### Dev and test

Training reads `dev.jsonl` and evaluates on `test.jsonl`. Declarations from the same pull request stay in one split, so a review's other edits cannot leak into the held-out set. Pull requests are shuffled with seed 1998 and assigned whole until the test set is as close as possible to 20% of the filtered triples.

| Split | Triples | Pull requests | Chosen shorter | Chosen longer | Same length |
| ----- | ------: | ------------: | -------------: | ------------: | ----------: |
| dev   | 431     | 283           | 347 (80.5%)    | 83 (19.3%)    | 1           |
| test  | 108     | 74            | 83 (76.9%)     | 24 (22.2%)    | 1           |

Regenerate both files with `python -m proof_quality.dataset_generation.split`. The training configs read them from `data/mathlibpr`.
