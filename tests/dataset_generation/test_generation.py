import json
import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from proof_quality.dataset_generation import (
    extract_pair,
    parse_hunks,
    parse_name_status,
)


GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "Test",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "Test",
    "GIT_COMMITTER_EMAIL": "test@example.com",
}


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        env=GIT_ENV,
        check=True,
    )
    return proc.stdout.strip()


def _write(repo: Path, name: str, text: str) -> None:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    _git(repo, "add", name)


def _commit(repo: Path, message: str) -> str:
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


BASE = """
variable {α : Type*}

theorem keepMe (x : α) : x = x := by
  rfl

theorem drift (n : Nat) : n = n := by
  rfl

theorem same (n : Nat) : n = n := by
  rfl

theorem hole : True := by
  sorry
""".lstrip()

EARLIER = """
variable {α : Type*}

theorem keepMe (x : α) : x = x := by
  apply rfl

/-- upstream doc -/
theorem drift (n : Nat) : n = n := by
  rfl

/-- same doc -/
theorem same (n : Nat) : n = n := by
  rfl

theorem hole : True := by
  sorry -- filled in later

theorem brand (k : Nat) : k = k := by
  apply rfl

theorem retarget (n : Nat) : n = n := by
  apply rfl

theorem note (n : Nat) : n = n := by
  rfl
""".lstrip()

BASE2 = """
variable {α : Type*}

theorem keepMe (x : α) : x = x := by
  rfl

theorem drift (n : Nat) : n = n := by
  exact rfl

theorem same (n : Nat) : n = n := by
  rfl

theorem hole : True := by
  sorry
""".lstrip()

FINAL = """
variable {α : Type*}

theorem keepMe (x : α) : x = x := by
  exact rfl

/-- upstream doc -/
theorem drift (n : Nat) : n = n := by
  exact rfl

/-- other doc -/
theorem same (n : Nat) : n = n := by
  rfl

theorem hole : True := by
  exact trivial

theorem brand (k : Nat) : k = k := by
  exact rfl

theorem retarget (n : Nat) : n = n ∧ True := by
  exact ⟨rfl, trivial⟩

theorem note (n : Nat) : n = n := by
  rfl -- explained
""".lstrip()


class DiffParsingTest(unittest.TestCase):
    def test_name_status_and_hunks(self) -> None:
        changes = parse_name_status(
            "M\tFoo.lean\nR100\tOld.lean\tNew.lean\nA\tBar.lean\n"
        )
        self.assertEqual([change.status for change in changes], ["M", "R", "A"])
        self.assertEqual(changes[1].old_path, "Old.lean")
        self.assertEqual(changes[1].new_path, "New.lean")
        hunks = parse_hunks("+++ b/Foo.lean\n@@ -10,0 +11,2 @@\n+a\n+b\n")
        self.assertEqual(hunks["Foo.lean"], [(11, 12)])


class ExtractPairTest(unittest.TestCase):
    def test_keeps_proof_edits_and_drops_the_rest(self) -> None:
        with TemporaryDirectory(prefix="mathlibpr-test-", dir=Path(".")) as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
            _write(repo, "File.lean", BASE)
            base1 = _commit(repo, "base1")
            _write(repo, "File.lean", EARLIER)
            earlier = _commit(repo, "earlier")
            _git(repo, "checkout", base1)
            _write(repo, "File.lean", BASE2)
            base2 = _commit(repo, "base2")
            _write(repo, "File.lean", FINAL)
            final = _commit(repo, "final")
            result = extract_pair(
                repo,
                {
                    "pair_id": "pr1",
                    "pr_number": 1,
                    "earlier_diff_base_sha": base1,
                    "earlier_snapshot_commit_sha": earlier,
                    "final_diff_base_sha": base2,
                    "final_snapshot_commit_sha": final,
                },
                "2026-10-22",
            )

        names = [row["metadata"]["declaration_name"] for row in result.rows]
        self.assertCountEqual(names, ["keepMe", "brand", "note"])
        kept = {row["metadata"]["declaration_name"]: row for row in result.rows}
        self.assertIn("{α : Type*}", kept["keepMe"]["prompt"])
        self.assertEqual(kept["keepMe"]["chosen"], "by\n  exact rfl")
        self.assertEqual(kept["keepMe"]["rejected"], "by\n  apply rfl")
        self.assertEqual(kept["note"]["chosen"], "by\n  rfl -- explained")
        self.assertEqual(kept["note"]["rejected"], "by\n  rfl")
        self.assertFalse(kept["keepMe"]["metadata"]["signature_changed"])
        self.assertEqual(kept["keepMe"]["metadata"]["rejection_reasons"], "")
        self.assertGreater(result.skips["unchanged_proof"], 0)
        self.assertGreater(result.skips["sorry"], 0)
        self.assertGreater(result.skips["upstream"], 0)
        self.assertEqual(result.skips["signature_changed"], 1)

    def test_pairs_a_renamed_file(self) -> None:
        with TemporaryDirectory(prefix="mathlibpr-test-", dir=Path(".")) as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
            _write(repo, "README.md", "root\n")
            root = _commit(repo, "root")
            _write(repo, "Old.lean", "theorem moved : True := by\n  apply trivial\n")
            earlier = _commit(repo, "earlier")
            _git(repo, "checkout", root)
            _write(repo, "New.lean", "theorem moved : True := by\n  exact trivial\n")
            final = _commit(repo, "final")
            result = extract_pair(
                repo,
                {
                    "pair_id": "pr2",
                    "pr_number": 2,
                    "earlier_diff_base_sha": root,
                    "earlier_snapshot_commit_sha": earlier,
                    "final_diff_base_sha": root,
                    "final_snapshot_commit_sha": final,
                },
                "2026-10-22",
            )
        self.assertEqual(
            [row["metadata"]["declaration_name"] for row in result.rows], ["moved"]
        )
        self.assertEqual(result.rows[0]["metadata"]["earlier_file_path"], "Old.lean")
        self.assertEqual(result.rows[0]["metadata"]["file_path"], "New.lean")


class RevertedProofTest(unittest.TestCase):
    def test_keeps_a_proof_restored_to_the_base(self) -> None:
        with TemporaryDirectory(prefix="mathlibpr-test-", dir=Path(".")) as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
            _write(
                repo,
                "Revert.lean",
                "theorem revertMe : True := by\n  rfl\n\ntheorem other : True := by\n  rfl\n",
            )
            _write(repo, "Drift.lean", "theorem drifted : True := by\n  rfl\n")
            _write(repo, "Gone.lean", "theorem gone : True := by\n  rfl\n")
            base = _commit(repo, "base")
            _write(
                repo,
                "Revert.lean",
                "theorem revertMe : True := by\n  apply rfl\n\ntheorem other : True := by\n  rfl\n",
            )
            _write(repo, "Drift.lean", "theorem drifted : True := by\n  apply rfl\n")
            _write(repo, "Gone.lean", "theorem gone : True := by\n  apply rfl\n")
            earlier = _commit(repo, "earlier")
            _git(repo, "checkout", base)
            _write(
                repo, "Drift.lean", "theorem drifted : True := by\n  exact trivial\n"
            )
            final_base = _commit(repo, "final-base")
            _write(
                repo,
                "Revert.lean",
                "theorem revertMe : True := by\n  rfl\n\ntheorem other : True := by\n  exact rfl\n",
            )
            _git(repo, "rm", "Gone.lean")
            final = _commit(repo, "final")
            result = extract_pair(
                repo,
                {
                    "pair_id": "pr3",
                    "pr_number": 3,
                    "earlier_diff_base_sha": base,
                    "earlier_snapshot_commit_sha": earlier,
                    "final_diff_base_sha": final_base,
                    "final_snapshot_commit_sha": final,
                },
                "2026-10-23",
            )

        names = [row["metadata"]["declaration_name"] for row in result.rows]
        self.assertEqual(names, ["revertMe"])
        self.assertEqual(result.rows[0]["rejected"], "by\n  apply rfl")
        self.assertEqual(result.rows[0]["chosen"], "by\n  rfl")
        self.assertTrue(result.rows[0]["metadata"]["proof_reverted"])
        self.assertEqual(result.skips["not_reverted"], 1)
        self.assertEqual(result.skips["one_sided"], 1)


class GronwallPilotTest(unittest.TestCase):
    def test_real_pr_23308_when_checkout_is_present(self) -> None:
        repo = Path("ignore/mathlib4")
        pairs = Path("ignore/within_pr_pairs_full.jsonl")
        if not (repo / ".git").exists() or not pairs.exists():
            self.skipTest("mathlib checkout or pairs file is not present")
        pair = None
        with pairs.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["pair_id"] == "pr23308":
                    pair = row
                    break
        self.assertIsNotNone(pair)
        result = extract_pair(repo, pair, "2026-10-22")
        self.assertIsNone(result.error, result.error)
        row = next(
            (
                item
                for item in result.rows
                if item["metadata"]["declaration_name"] == "ODE_solution_unique_univ"
            ),
            None,
        )
        self.assertIsNotNone(row)
        assert row is not None
        self.assertIn("apply lt_of_lt_of_le", row["rejected"])
        self.assertIn("exact sub_one_lt", row["chosen"])
        self.assertFalse(row["metadata"]["signature_changed"])
        self.assertIn("{E : Type*}", row["prompt"])
        self.assertNotIn("εf", row["prompt"])


if __name__ == "__main__":
    unittest.main()
