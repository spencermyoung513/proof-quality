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




### On length bias

A natural / "dumb" heuristic for a proof that gets accepted into Mathlib is "the shorter, the better". Indeed, this is often the case when a more elegant set of tactics may achieve the same goal. However, we also want to capture the cases where this rule doesn't hold, since there are other quality signals in a proof besides length. Our dataset, while not perfectly balanced, represents both cases well: there are 1,638 triplets (71.9%) where the chosen proof is shorter than the rejected one, while the other 640 (28.1%) have a rejected proof that is shorter than the chosen. 192 of the triplets are edits the accepted snapshot restored, marked with `metadata.proof_reverted`. On that slice the two directions are close: the chosen proof is shorter in 89, and the rejected proof is shorter in 101.
