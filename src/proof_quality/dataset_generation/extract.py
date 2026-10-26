"""Extract theorem-proof preference pairs from MathlibPR snapshots.

The published ``pairs`` rows only link two commits. This module reads those
commits from a mathlib4 checkout, keeps theorems and lemmas whose proof
changed and whose statement did not, between the first build-success
snapshot and the accepted snapshot. A proof edited in the rejected snapshot
and restored to its previous text in the accepted snapshot is kept too.
The rows are the prompt/chosen/rejected examples the reward model trains on.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from proof_quality.dataset_generation.lean_source import (
    Declaration,
    is_sorry_proof,
    comparable_proof,
    normalize_whitespace,
    parse_lean_file,
    proof_text,
)


PROMPT = (
    "Write a Mathlib-quality Lean 4 proof of the following declaration.\n\n"
    "{signature} :="
)
HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
DATASET_VERSION = "mathlibpr-pilot"


@dataclass(frozen=True)
class FileChange:
    status: str
    old_path: str | None
    new_path: str | None


@dataclass
class PairResult:
    rows: list[dict] = field(default_factory=list)
    skips: Counter = field(default_factory=Counter)
    lean_files: int = 0
    error: str | None = None


def load_pairs(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def select_pairs(rows: list[dict], pr_numbers: list[int], limit: int) -> list[dict]:
    """Pin requested PRs, then fill from the file until ``limit`` rows."""
    by_pr = {int(row["pr_number"]): row for row in rows}
    selected: list[dict] = []
    seen: set[int] = set()
    for number in pr_numbers:
        row = by_pr.get(number)
        if row is None:
            raise SystemExit(f"PR {number} is not in the pairs file")
        selected.append(row)
        seen.add(number)
    if not limit and not pr_numbers:
        return rows
    if limit and len(selected) >= limit and pr_numbers:
        return selected
    if not limit:
        return selected
    for row in rows:
        number = int(row["pr_number"])
        if number in seen:
            continue
        selected.append(row)
        seen.add(number)
        if len(selected) >= limit:
            break
    return selected


def parse_name_status(text: str) -> list[FileChange]:
    changes: list[FileChange] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        kind = parts[0][0]
        if kind in {"R", "C"} and len(parts) >= 3:
            changes.append(FileChange(kind, parts[1], parts[2]))
        elif kind == "A" and len(parts) >= 2:
            changes.append(FileChange("A", None, parts[1]))
        elif kind == "M" and len(parts) >= 2:
            changes.append(FileChange("M", parts[1], parts[1]))
        elif kind == "D" and len(parts) >= 2:
            changes.append(FileChange("D", parts[1], None))
    return changes


def parse_hunks(diff_text: str) -> dict[str, list[tuple[int, int]]]:
    """Map a new-file path to inclusive 1-based ranges added or edited there."""
    hunks: dict[str, list[tuple[int, int]]] = {}
    current: str | None = None
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            path = line[4:]
            if path.startswith("b/"):
                path = path[2:]
            current = None if path == "/dev/null" else path
            if current is not None:
                hunks.setdefault(current, [])
            continue
        if current is None:
            continue
        match = HUNK_RE.match(line)
        if match is None:
            continue
        start = int(match.group(1))
        length = int(match.group(2) or "1")
        if length <= 0:
            continue
        hunks[current].append((start, start + length - 1))
    return hunks


def extract_pair(
    repo: Path, pair: dict, extracted_on: str, dataset_version: str = DATASET_VERSION
) -> PairResult:
    """Return preference rows for one within-PR snapshot pair."""
    result = PairResult()
    try:
        earlier_changes = _lean_changes(
            repo, pair["earlier_diff_base_sha"], pair["earlier_snapshot_commit_sha"]
        )
        final_changes = _lean_changes(
            repo, pair["final_diff_base_sha"], pair["final_snapshot_commit_sha"]
        )
        earlier_hunks = parse_hunks(
            _git_diff(
                repo, pair["earlier_diff_base_sha"], pair["earlier_snapshot_commit_sha"]
            )
        )
        final_hunks = parse_hunks(
            _git_diff(
                repo, pair["final_diff_base_sha"], pair["final_snapshot_commit_sha"]
            )
        )
    except subprocess.CalledProcessError as exc:
        result.error = (exc.stderr or exc.stdout or "git failed").strip()
        result.skips["git_error"] += 1
        return result

    result.lean_files = len(
        {
            change.new_path
            for change in earlier_changes + final_changes
            if change.new_path
        }
    )
    earlier = _touched(
        repo, pair["earlier_snapshot_commit_sha"], earlier_changes, earlier_hunks
    )
    final = _touched(
        repo, pair["final_snapshot_commit_sha"], final_changes, final_hunks
    )
    earlier_decls, earlier_ambiguous = earlier
    final_decls, final_ambiguous = final
    for name in earlier_ambiguous | final_ambiguous:
        result.skips["ambiguous_name"] += 1
        earlier_decls.pop(name, None)
        final_decls.pop(name, None)

    for name, (earlier_decl, earlier_path) in earlier_decls.items():
        final_hit = final_decls.get(name)
        reverted = False
        if final_hit is None:
            recovered = _recover_final_declaration(
                repo, pair["final_snapshot_commit_sha"], earlier_path, name
            )
            if isinstance(recovered, str):
                result.skips[recovered] += 1
                continue
            final_decl, final_path = recovered
            reverted = True
        else:
            final_decl, final_path = final_hit
        if is_sorry_proof(earlier_decl.body) or is_sorry_proof(final_decl.body):
            result.skips["sorry"] += 1
            continue
        if comparable_proof(earlier_decl.body) == comparable_proof(final_decl.body):
            result.skips["unchanged_proof"] += 1
            continue
        earlier_base = _base_body(
            repo,
            pair["earlier_diff_base_sha"],
            earlier_changes,
            earlier_path,
            name,
        )
        final_base = _base_body(
            repo, pair["final_diff_base_sha"], final_changes, final_path, name
        )
        if _is_upstream(earlier_decl.body, final_decl.body, earlier_base, final_base):
            result.skips["upstream"] += 1
            continue
        earlier_signature, _ = earlier_decl.rendered()
        final_signature, _ = final_decl.rendered()
        if normalize_whitespace(earlier_signature) != normalize_whitespace(
            final_signature
        ):
            result.skips["signature_changed"] += 1
            continue
        if reverted and not _is_reverted_proof(
            earlier_decl.body, final_decl.body, earlier_base
        ):
            earlier_matches_base = earlier_base is not None and comparable_proof(
                earlier_decl.body
            ) == comparable_proof(earlier_base)
            reason = "upstream" if earlier_matches_base else "not_reverted"
            result.skips[reason] += 1
            continue
        result.rows.append(
            _row(
                pair,
                earlier_decl,
                earlier_path,
                final_decl,
                final_path,
                extracted_on,
                dataset_version,
                reverted=reverted,
            )
        )
    return result


def missing_commits(repo: Path, shas: list[str]) -> list[str]:
    unique = list(dict.fromkeys(shas))
    if not unique:
        return []
    env = {**os.environ, "GIT_NO_LAZY_FETCH": "1"}
    proc = subprocess.run(
        ["git", "cat-file", "--batch-check"],
        cwd=repo,
        input="".join(f"{sha}^{{commit}}\n" for sha in unique),
        text=True,
        capture_output=True,
        env=env,
    )
    missing: list[str] = []
    for sha, line in zip(unique, proc.stdout.splitlines()):
        if line.endswith(" missing"):
            missing.append(sha)
    return missing


def fetch_commits(repo: Path, shas: list[str]) -> list[str]:
    """Fetch missing commits. Returns SHAs that origin would not provide."""
    missing = missing_commits(repo, shas)
    failed: list[str] = []
    for start in range(0, len(missing), 40):
        batch = missing[start : start + 40]
        print(f"fetch {start + len(batch)}/{len(missing)}", flush=True)
        if _fetch_batch(repo, batch):
            continue
        for sha in batch:
            if not _fetch_batch(repo, [sha]):
                failed.append(sha)
    return failed


def _fetch_batch(repo: Path, shas: list[str]) -> bool:
    proc = subprocess.run(
        ["git", "fetch", "--filter=blob:none", "origin", *shas],
        cwd=repo,
        text=True,
        capture_output=True,
    )
    return proc.returncode == 0


def _recover_final_declaration(
    repo: Path, sha: str, path: str, qualified_name: str
) -> tuple[Declaration, str] | str:
    """Find a declaration the accepted diff did not touch.

    Returns the declaration and path, or a skip reason when it is not a
    unique declaration in that file at the accepted snapshot.
    """
    text = _read_file(repo, sha, path)
    if text is None:
        return "one_sided"
    matches = [
        decl for decl in parse_lean_file(text) if decl.qualified_name == qualified_name
    ]
    if len(matches) > 1:
        return "ambiguous_name"
    if len(matches) != 1:
        return "one_sided"
    return matches[0], path


def _is_reverted_proof(
    earlier_body: str, final_body: str, earlier_base: str | None
) -> bool:
    """The accepted proof is the proof from before the rejected edit."""
    if earlier_base is None:
        return False
    base = comparable_proof(earlier_base)
    return (
        comparable_proof(final_body) == base and comparable_proof(earlier_body) != base
    )


def _row(
    pair: dict,
    earlier: Declaration,
    earlier_path: str,
    final: Declaration,
    final_path: str,
    extracted_on: str,
    dataset_version: str,
    reverted: bool = False,
) -> dict:
    earlier_signature, _ = earlier.rendered()
    final_signature, _ = final.rendered()
    rejected = proof_text(earlier.body)
    chosen = proof_text(final.body)
    initial_lines = len(rejected.splitlines())
    final_lines = len(chosen.splitlines())
    pr_number = int(pair["pr_number"])
    return {
        "prompt": PROMPT.format(signature=final_signature),
        "chosen": chosen,
        "rejected": rejected,
        "metadata": {
            "pr_number": pr_number,
            "pr_url": f"https://github.com/leanprover-community/mathlib4/pull/{pr_number}",
            "pair_id": pair["pair_id"],
            "declaration_name": final.qualified_name,
            "declaration_kind": final.kind,
            "file_path": final_path,
            "earlier_file_path": earlier_path,
            "earlier_snapshot_commit_sha": pair["earlier_snapshot_commit_sha"],
            "final_snapshot_commit_sha": pair["final_snapshot_commit_sha"],
            "earlier_diff_base_sha": pair["earlier_diff_base_sha"],
            "final_diff_base_sha": pair["final_diff_base_sha"],
            "signature_changed": normalize_whitespace(earlier_signature)
            != normalize_whitespace(final_signature),
            "initial_line_count": initial_lines,
            "final_line_count": final_lines,
            "line_count_delta": final_lines - initial_lines,
            "proof_reverted": reverted,
            "rejection_reasons": "",
            "feedback_categories": [],
            "key_changes": [],
            "dataset_version": dataset_version,
            "extraction_date": extracted_on,
        },
    }


def _is_upstream(
    earlier_body: str,
    final_body: str,
    earlier_base: str | None,
    final_base: str | None,
) -> bool:
    if earlier_base is None or final_base is None:
        return False
    return comparable_proof(earlier_body) == comparable_proof(
        earlier_base
    ) and comparable_proof(final_body) == comparable_proof(final_base)


def _base_body(
    repo: Path,
    base_sha: str,
    changes: list[FileChange],
    snapshot_path: str,
    qualified_name: str,
) -> str | None:
    old_path = next(
        (change.old_path for change in changes if change.new_path == snapshot_path),
        None,
    )
    if not old_path:
        return None
    text = _read_file(repo, base_sha, old_path)
    if text is None:
        return None
    matches = [
        decl.body
        for decl in parse_lean_file(text)
        if decl.qualified_name == qualified_name
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _touched(
    repo: Path,
    sha: str,
    changes: list[FileChange],
    hunks: dict[str, list[tuple[int, int]]],
) -> tuple[dict[str, tuple[Declaration, str]], set[str]]:
    found: dict[str, tuple[Declaration, str]] = {}
    ambiguous: set[str] = set()
    for change in changes:
        if change.new_path is None:
            continue
        text = _read_file(repo, sha, change.new_path)
        if text is None:
            continue
        ranges = hunks.get(change.new_path, [])
        if change.status == "A" and not ranges:
            ranges = [(1, max(1, len(text.splitlines())))]
        for decl in parse_lean_file(text):
            if not _overlaps(decl, ranges):
                continue
            if decl.qualified_name in ambiguous or decl.qualified_name in found:
                ambiguous.add(decl.qualified_name)
                found.pop(decl.qualified_name, None)
                continue
            found[decl.qualified_name] = (decl, change.new_path)
    return found, ambiguous


def _overlaps(decl: Declaration, ranges: list[tuple[int, int]]) -> bool:
    return any(
        decl.start_line <= end and start <= decl.end_line for start, end in ranges
    )


def _lean_changes(repo: Path, base: str, sha: str) -> list[FileChange]:
    text = _git(
        repo,
        "diff",
        "--name-status",
        "--find-renames",
        base,
        sha,
        "--",
        "*.lean",
    )
    return parse_name_status(text)


def _git_diff(repo: Path, base: str, sha: str) -> str:
    return _git(
        repo,
        "diff",
        "-U0",
        "--no-color",
        "--find-renames",
        base,
        sha,
        "--",
        "*.lean",
    )


def _read_file(repo: Path, sha: str, path: str) -> str | None:
    proc = subprocess.run(
        ["git", "-c", "core.quotepath=false", "show", f"{sha}:{path}"],
        cwd=repo,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-c", "core.quotepath=false", *args],
        cwd=repo,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, proc.args, proc.stdout, proc.stderr
        )
    return proc.stdout


def _pair_shas(pair: dict) -> list[str]:
    return [
        pair["earlier_snapshot_commit_sha"],
        pair["earlier_diff_base_sha"],
        pair["final_snapshot_commit_sha"],
        pair["final_diff_base_sha"],
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m proof_quality.mathlibpr", description=__doc__
    )
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--pr", type=int, action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch missing snapshot commits from origin before extracting",
    )
    parser.add_argument("--dataset-version", default=DATASET_VERSION)
    args = parser.parse_args(argv)

    pairs = select_pairs(load_pairs(args.pairs), args.pr, args.limit)
    wanted = [sha for pair in pairs for sha in _pair_shas(pair)]
    if args.fetch:
        fetch_commits(args.repo, wanted)
    missing = set(missing_commits(args.repo, wanted))
    if missing and not args.fetch:
        preview = ", ".join(list(missing)[:4])
        raise SystemExit(
            f"{len(missing)} commits are not in {args.repo} ({preview}). "
            "Re-run with --fetch."
        )

    extracted_on = date.today().isoformat()
    skips: Counter = Counter()
    errors: list[dict] = []
    file_counts: list[dict] = []
    written = 0
    signature_changed = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary or args.output.with_suffix(".summary.json")

    def write_summary() -> None:
        summary = {
            "prs_scanned": len(file_counts) + skips["missing_commit"],
            "prs_with_pairs": sum(1 for item in file_counts if item["kept"]),
            "declarations_emitted": written,
            "signature_changed": signature_changed,
            "skips": dict(skips),
            "errors": errors,
            "files": file_counts,
            "output": str(args.output),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    with args.output.open("w", encoding="utf-8") as handle:
        for index, pair in enumerate(pairs, start=1):
            if missing.intersection(_pair_shas(pair)):
                skips["missing_commit"] += 1
                errors.append(
                    {"pair_id": pair["pair_id"], "error": "missing snapshot commit"}
                )
                print(f"{pair['pair_id']}: missing commit", flush=True)
                continue
            result = extract_pair(args.repo, pair, extracted_on, args.dataset_version)
            for row in result.rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                if row["metadata"]["signature_changed"]:
                    signature_changed += 1
            handle.flush()
            skips.update(result.skips)
            file_counts.append(
                {
                    "pair_id": pair["pair_id"],
                    "pr_number": int(pair["pr_number"]),
                    "lean_files": result.lean_files,
                    "kept": len(result.rows),
                }
            )
            if result.error:
                errors.append({"pair_id": pair["pair_id"], "error": result.error})
            print(
                f"{pair['pair_id']}: files={result.lean_files} kept={len(result.rows)}",
                flush=True,
            )
            if index % 100 == 0:
                print(
                    f"progress {index}/{len(pairs)} kept={written}",
                    flush=True,
                )
                write_summary()
    write_summary()
    print(
        f"Wrote {written} declarations from {len(pairs)} PRs to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
