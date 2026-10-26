import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from tempfile import TemporaryDirectory
from pathlib import Path

from proof_quality.dataset_generation.split import main, split_rows


def _row(pr: int, name: str) -> dict:
    return {
        "prompt": name,
        "chosen": "by simp",
        "rejected": "by grind",
        "metadata": {"pr_number": pr, "declaration_name": name},
    }


class SplitTests(unittest.TestCase):
    def test_a_pull_request_stays_in_one_split(self) -> None:
        rows = [_row(pr, f"d{pr}-{i}") for pr in range(30) for i in range(pr % 3 + 1)]
        dev, test = split_rows(rows, test_fraction=0.2, seed=1998)
        dev_prs = {row["metadata"]["pr_number"] for row in dev}
        test_prs = {row["metadata"]["pr_number"] for row in test}
        self.assertTrue(dev_prs.isdisjoint(test_prs))
        self.assertEqual(
            dev_prs | test_prs, {row["metadata"]["pr_number"] for row in rows}
        )
        source = [row["metadata"]["declaration_name"] for row in rows]
        for part in (dev, test):
            positions = [
                source.index(row["metadata"]["declaration_name"]) for row in part
            ]
            self.assertEqual(positions, sorted(positions))
        self.assertTrue(all(row["metadata"]["split"] == "dev" for row in dev))
        self.assertTrue(all(row["metadata"]["split"] == "test" for row in test))
        self.assertLess(abs(len(test) / len(rows) - 0.2), 0.05)

    def test_the_same_seed_is_stable(self) -> None:
        rows = [_row(pr, f"d{pr}") for pr in range(40)]
        self.assertEqual(split_rows(rows, seed=1998), split_rows(rows, seed=1998))

    def test_command_writes_both_files(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "pairs.substantive.jsonl"
            rows = [_row(pr, f"d{pr}") for pr in range(20)]
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            dev_path = root / "dev.jsonl"
            test_path = root / "test.jsonl"
            with redirect_stdout(StringIO()):
                main(
                    [
                        "--input",
                        str(source),
                        "--dev",
                        str(dev_path),
                        "--test",
                        str(test_path),
                        "--seed",
                        "1998",
                    ]
                )
            dev = [json.loads(line) for line in dev_path.read_text().splitlines()]
            test = [json.loads(line) for line in test_path.read_text().splitlines()]
            self.assertEqual(len(dev) + len(test), len(rows))
            self.assertTrue(test)
            self.assertTrue(dev)


if __name__ == "__main__":
    unittest.main()
