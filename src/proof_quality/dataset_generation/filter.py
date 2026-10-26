"""Filter Mathlib convention edits out of a proof-pair file.

``python -m proof_quality.dataset_generation.filter`` reads ``pairs.jsonl`` and
writes the kept rows, an annotated copy of every row, and a count section in
the dataset README.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from proof_quality.dataset_generation.substantive import KEEP_CLASSES, proof_diff_class


_REPO = Path(__file__).resolve().parents[3]
_DATA = _REPO / "data" / "mathlibpr"
_HEADING = "## Filtering convention edits"
_LENGTH_HEADING = "### On length bias"

_CLASS_NOTES = {
    "construction": (
        "The script gains or loses a real step, or a hammer is swapped for a written proof."
    ),
    "comment": "The Lean code matches, and the comment wording itself changes.",
    "comment_spacing": (
        "The code matches, and the comments differ only by spacing or reflow."
    ),
    "style": "Tactic synonyms, `@`, pipes, or `by exact` versus a bare term.",
    "layout": "Parentheses, bullets, or other punctuation.",
    "rename": (
        "An identifier or namespace edit, including one lemma name replaced in place."
    ),
    "simp_list": "Lemmas added or removed inside one `simp` or `rw` list.",
    "bookkeeping": (
        "Binders, holes, projections, or explicit arguments on the same proof."
    ),
    "minor": "A non-structural tweak, such as deleting a redundant `rw`.",
}


def with_diff_class(row: dict) -> dict:
    """Return ``row`` with ``metadata.diff_class`` set."""
    metadata = dict(row["metadata"])
    metadata["diff_class"] = proof_diff_class(row["chosen"], row["rejected"])
    return {**row, "metadata": metadata}


def readme_section(counts: Counter[str], total: int) -> str:
    """Describe the filter rules and how many rows each class holds."""
    kept = sum(counts[name] for name in KEEP_CLASSES)
    lines = [
        _HEADING,
        "",
        "Many of the triples in `pairs.jsonl` differ because a revision followed a Mathlib convention, not because the two proofs are different arguments. The filter compares each pair after separating comments from code and normalizing those conventions. It keeps a triple when the proof script changes shape, or when the code matches and the comment wording itself changes.",
        "",
        "A script changes shape when a real step appears or disappears: `calc`, `induction`, `cases`, `ext`, or a `have` or `let` whose body is itself a proof. A hammer such as `grind` or `omega` swapped for a written proof counts too. `aesop` and `cat_disch` are the same discharger. The same tactics aimed at a different lemma do not count, and neither does a `have` or `let` that only names a term or an instance. Comment edits that only reflow text, including `n + 1` versus `n+1`, are dropped. A convention change in the code is still dropped when a comment changes along with it.",
        "",
        "What gets dropped is the convention layer: renames and coercions (`(n : ℕ)` versus `n`, `eq_zero_or_pos` versus `eq_or_ne`), lemmas moved inside one `simp` or `rw` list, tactic synonyms (`apply`/`exact`, `field_simp`/`simp`, `obtain`/`cases`, `induction'` versus structured `induction`), reordered bullets, layout (`@`, pipes, parentheses, `by exact` versus a bare term), and the same proof with new binders, holes, projections, or explicit arguments. Folding a term-mode `have` into `simp`, writing `#s` for `s.card`, or replacing borel boilerplate with `borelize` is the same kind of edit.",
        "",
        f"`pairs.substantive.jsonl` is the kept subset ({kept:,} of {total:,}). `pairs.annotated.jsonl` labels every triple with `metadata.diff_class`. Regenerate both with `python -m proof_quality.dataset_generation.filter`.",
        "",
        "| Class | Rows | Share | Verdict | What it is |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    order = [name for name, _note in _CLASS_NOTES.items() if counts[name]]
    order += sorted(name for name in counts if name not in _CLASS_NOTES)
    for name in order:
        count = counts[name]
        verdict = "keep" if name in KEEP_CLASSES else "drop"
        note = _CLASS_NOTES.get(name, "")
        share = f"{count / total:.1%}" if total else "0%"
        lines.append(f"| `{name}` | {count:,} | {share} | {verdict} | {note} |")
    lines.append("")
    return "\n".join(lines)


def write_readme(path: Path, section: str) -> None:
    """Write ``section`` before the length-bias note, replacing an older copy."""
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    prefix, length_note = _split_length_note(existing)
    if _HEADING in prefix:
        prefix = prefix[: prefix.index(_HEADING)]
    text = prefix.rstrip() + "\n\n" + section.rstrip()
    if length_note:
        text += "\n\n" + length_note.rstrip()
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _split_length_note(text: str) -> tuple[str, str]:
    """Separate the length-bias note from the rest of the README."""
    index = text.find(_LENGTH_HEADING)
    if index < 0:
        return text, ""
    note = text[index:]
    if _HEADING in note:
        note = note[: note.index(_HEADING)]
    return text[:index], note.strip()


def filter_pairs(
    source: Path,
    output: Path,
    annotated: Path,
    readme: Path | None,
) -> Counter[str]:
    """Classify ``source`` and write the kept and annotated files."""
    counts: Counter[str] = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    annotated.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with (
        source.open(encoding="utf-8") as incoming,
        output.open("w", encoding="utf-8") as kept_handle,
        annotated.open("w", encoding="utf-8") as annotated_handle,
    ):
        for line in incoming:
            if not line.strip():
                continue
            row = with_diff_class(json.loads(line))
            total += 1
            diff_class = row["metadata"]["diff_class"]
            counts[diff_class] += 1
            encoded = json.dumps(row, ensure_ascii=False) + "\n"
            annotated_handle.write(encoded)
            if diff_class in KEEP_CLASSES:
                kept_handle.write(encoded)
    if readme is not None:
        write_readme(readme, readme_section(counts, total))
    return counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m proof_quality.dataset_generation.filter",
        description=__doc__,
    )
    parser.add_argument("--input", type=Path, default=_DATA / "pairs.jsonl")
    parser.add_argument(
        "--output", type=Path, default=_DATA / "pairs.substantive.jsonl"
    )
    parser.add_argument(
        "--annotated", type=Path, default=_DATA / "pairs.annotated.jsonl"
    )
    parser.add_argument("--readme", type=Path, default=_DATA / "README.md")
    parser.add_argument(
        "--no-readme",
        action="store_true",
        help="Do not write the count section into the dataset README",
    )
    args = parser.parse_args(argv)
    counts = filter_pairs(
        args.input,
        args.output,
        args.annotated,
        None if args.no_readme else args.readme,
    )
    total = sum(counts.values())
    kept = sum(counts[name] for name in KEEP_CLASSES)
    print(f"kept {kept} of {total}")
    for name, count in counts.most_common():
        verdict = "keep" if name in KEEP_CLASSES else "drop"
        print(f"  {name}: {count} ({verdict})")


if __name__ == "__main__":
    main()
