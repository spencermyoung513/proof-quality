"""Split filtered proof pairs into a dev set and a held-out test set.

``python -m proof_quality.dataset_generation.split`` reads
``pairs.substantive.jsonl`` and writes ``dev.jsonl`` and ``test.jsonl``.
Training loads ``dev``; evaluation loads ``test``. Every declaration from one
pull request stays in the same split.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


_REPO = Path(__file__).resolve().parents[3]
_DATA = _REPO / "data" / "mathlibpr"
DEFAULT_SEED = 1998
DEFAULT_TEST_FRACTION = 0.2


def split_rows(
    rows: list[dict],
    *,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    seed: int = DEFAULT_SEED,
) -> tuple[list[dict], list[dict]]:
    """Return ``(dev, test)`` with ``metadata.split`` set on each row.

    Pull requests are shuffled with ``seed`` and assigned whole. A request
    joins the test set only when that moves the test count closer to
    ``test_fraction`` of the rows.
    """
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be between 0 and 1, got {test_fraction}")
    groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[int(row["metadata"]["pr_number"])].append(index)
    test_prs = _choose_test_prs(groups, len(rows), test_fraction, seed)
    dev: list[dict] = []
    test: list[dict] = []
    for row in rows:
        split = "test" if int(row["metadata"]["pr_number"]) in test_prs else "dev"
        labeled = {**row, "metadata": {**row["metadata"], "split": split}}
        (test if split == "test" else dev).append(labeled)
    return dev, test


def _choose_test_prs(
    groups: dict[int, list[int]],
    total: int,
    test_fraction: float,
    seed: int,
) -> set[int]:
    target = round(total * test_fraction)
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    chosen: list[int] = []
    count = 0
    for pr in order:
        size = len(groups[pr])
        if abs(count + size - target) < abs(count - target):
            chosen.append(pr)
            count += size
    return set(chosen)


def write_split(
    dev: list[dict], test: list[dict], dev_path: Path, test_path: Path
) -> None:
    """Write each split as JSONL, in the input row order."""
    for path, rows in ((dev_path, dev), (test_path, test)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m proof_quality.dataset_generation.split",
        description=__doc__,
    )
    parser.add_argument("--input", type=Path, default=_DATA / "pairs.substantive.jsonl")
    parser.add_argument("--dev", type=Path, default=_DATA / "dev.jsonl")
    parser.add_argument("--test", type=Path, default=_DATA / "test.jsonl")
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dev, test = split_rows(rows, test_fraction=args.test_fraction, seed=args.seed)
    write_split(dev, test, args.dev, args.test)
    print(f"dev {len(dev)}  test {len(test)}  seed {args.seed}")


if __name__ == "__main__":
    main()
