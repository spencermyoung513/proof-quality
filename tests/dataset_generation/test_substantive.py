import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from proof_quality.dataset_generation.filter import main
from proof_quality.dataset_generation.substantive import KEEP_CLASSES, proof_diff_class


def _kept(chosen: str, rejected: str) -> bool:
    return proof_diff_class(chosen, rejected) in KEEP_CLASSES


class SubstantiveDiffTests(unittest.TestCase):
    def test_comment_spacing_is_dropped(self) -> None:
        rejected = "exact foo -- g(t) = (x - t)^(n + 1)"
        chosen = "exact foo -- g(t) = (x - t)^(n+1)"
        self.assertEqual(proof_diff_class(chosen, rejected), "comment_spacing")
        self.assertFalse(_kept(chosen, rejected))

    def test_rewritten_comment_is_kept(self) -> None:
        rejected = "exact h -- base case\nexact h"
        chosen = "exact h\nexact h"
        self.assertEqual(proof_diff_class(chosen, rejected), "comment")
        self.assertTrue(_kept(chosen, rejected))

    def test_comment_notation_rewrite_is_kept(self) -> None:
        rejected = "exact h -- let `a_inv` be the inverse"
        chosen = "exact h -- let `(f a)⁻¹` be the inverse"
        self.assertEqual(proof_diff_class(chosen, rejected), "comment")

    def test_style_edit_drops_even_if_the_comment_changes(self) -> None:
        rejected = "apply foo -- the reviewer asked for a different argument"
        chosen = "exact foo -- this sentence is a new explanation of the step"
        self.assertEqual(proof_diff_class(chosen, rejected), "style")
        self.assertFalse(_kept(chosen, rejected))

    def test_explicit_argument_marker_is_dropped(self) -> None:
        rejected = "| @insert i I hi_nmem_I h_ind =>"
        chosen = "| insert i I hi_nmem_I h_ind =>"
        self.assertEqual(proof_diff_class(chosen, rejected), "style")

    def test_pipe_versus_apply_is_dropped(self) -> None:
        rejected = (
            "· apply lt_of_lt_of_le (sub_one_lt _) "
            "(le_trans (min_le_left _ _) (neg_abs_le t))"
        )
        chosen = (
            "· exact sub_one_lt _ |>.trans_le <| "
            "min_le_left _ _ |>.trans <| neg_abs_le t"
        )
        self.assertEqual(proof_diff_class(chosen, rejected), "rename")
        self.assertFalse(_kept(chosen, rejected))

    def test_reassociated_trans_chain_is_dropped(self) -> None:
        rejected = (
            "apply lt_of_le_of_lt (le_trans (le_abs_self _) (le_max_left _ _)) "
            "(lt_add_one _)"
        )
        chosen = "exact le_abs_self _ |>.trans <| le_max_left _ _ |>.trans_lt <| lt_add_one _"
        self.assertEqual(proof_diff_class(chosen, rejected), "style")
        self.assertFalse(_kept(chosen, rejected))

    def test_bare_term_matches_by_exact(self) -> None:
        self.assertEqual(
            proof_diff_class(
                "hasDerivAtFilter_const _ _ _",
                "by exact hasDerivAtFilter_const _ _ _",
            ),
            "style",
        )

    def test_field_simp_is_a_simp_list_edit(self) -> None:
        self.assertEqual(proof_diff_class("simp [field]", "field_simp"), "simp_list")
        self.assertEqual(
            proof_diff_class("simp [field, c]", "field_simp [c]"),
            "simp_list",
        )

    def test_namespace_qualification_is_a_rename(self) -> None:
        rejected = "exact Set.Rel.not_infiniteDimensional_iff"
        chosen = "exact SetRel.not_infiniteDimensional_iff"
        self.assertEqual(proof_diff_class(chosen, rejected), "rename")

    def test_single_unrelated_identifier_is_a_rename(self) -> None:
        rejected = "exact div_modEq_inj"
        chosen = "exact ext_div_modEq_iff"
        self.assertEqual(proof_diff_class(chosen, rejected), "rename")

    def test_hammer_synonym_is_dropped(self) -> None:
        self.assertEqual(proof_diff_class("by cat_disch", "by aesop"), "style")

    def test_parentheses_only_is_layout(self) -> None:
        self.assertEqual(proof_diff_class("exact (foo)", "exact foo"), "layout")

    def test_redundant_rewrite_is_minor(self) -> None:
        rejected = "rw [h]\nexact goal"
        chosen = "exact goal"
        self.assertEqual(proof_diff_class(chosen, rejected), "minor")
        self.assertFalse(_kept(chosen, rejected))

    def test_hammer_versus_written_proof_is_kept(self) -> None:
        self.assertEqual(
            proof_diff_class("cases x <;> simp", "by grind"),
            "construction",
        )
        self.assertTrue(_kept("cases x <;> simp", "by grind"))

    def test_calc_block_versus_one_lemma_is_kept(self) -> None:
        rejected = """by
  have h : a = b := by rw [ha]
  calc
    a = b := h
    _ = c := hc
"""
        chosen = "finished ha hc"
        self.assertEqual(proof_diff_class(chosen, rejected), "construction")

    def test_same_tactics_with_a_new_argument_are_not_kept(self) -> None:
        rejected = "rw [AffineMap.apply_lineMap]\nsimp only [vsub_eq_sub]"
        chosen = "trans c\n· simp [lineMap_apply_ring']\n· rw [sub_mul]"
        self.assertNotIn(proof_diff_class(chosen, rejected), KEEP_CLASSES)

    def test_induction_binder_names_are_bookkeeping(self) -> None:
        rejected = """by
  classical
  induction s using Finset.induction with
  | empty => simp
  | insert has ih => simpa [has, ← ih] using emultiplicity_mul hp
"""
        chosen = """by classical
  induction s using Finset.induction with
  | empty => simp
  | insert a s has ih => simpa [has, ← ih] using emultiplicity_mul hp
"""
        self.assertEqual(proof_diff_class(chosen, rejected), "bookkeeping")

    def test_dropped_argument_and_hole_are_bookkeeping(self) -> None:
        rejected = "fun z hz ↦ (hf z hz).meromorphicAt"
        chosen = "fun _ hz ↦ (hf hz).meromorphicAt"
        self.assertEqual(proof_diff_class(chosen, rejected), "bookkeeping")

    def test_projection_and_explicit_argument_are_bookkeeping(self) -> None:
        self.assertEqual(
            proof_diff_class(
                "(isColoop_iff_forall_mem_closure_iff_mem.1 he) X",
                "(isColoop_iff_forall_mem_closure_iff_mem.1 he).1 X",
            ),
            "bookkeeping",
        )
        self.assertEqual(
            proof_diff_class(
                "DilationEquiv.map_cobounded (DilationEquiv.mulLeft a ha)",
                "DilationEquiv.map_cobounded (DilationEquiv.mulLeft ha)",
            ),
            "bookkeeping",
        )

    def test_card_notation_is_style(self) -> None:
        self.assertEqual(
            proof_diff_class("refine ⟨#s, c, ?_⟩", "refine ⟨s.card, c, ?_⟩"),
            "style",
        )

    def test_inlined_have_is_style(self) -> None:
        rejected = """by
  rw [← range_gluedTo, I.subschemeι_def]
  have : Set.range I.subschemeIso.hom.base = .univ :=
    Set.range_eq_univ.mpr I.subschemeIso.hom.homeomorph.surjective
  simp [Set.range_comp, this]
"""
        chosen = """by
  simp [← range_gluedTo, I.subschemeι_def, Set.range_comp,
    Set.range_eq_univ.mpr I.subschemeIso.hom.homeomorph.surjective]
"""
        self.assertEqual(proof_diff_class(chosen, rejected), "style")

    def test_inlined_by_and_borelize_are_not_kept(self) -> None:
        self.assertFalse(
            _kept(
                "exact isOpen_setOf_linearIndependent.preimage (by fun_prop)",
                "refine isOpen_setOf_linearIndependent.preimage ?_\nfun_prop",
            )
        )
        rejected = """by
  obtain ⟨rfl⟩ := ‹BorelSpace A›
  letI := borel A
  exact addHaarScalarFactor_smul_congr' ..
"""
        chosen = """by
  borelize A
  exact addHaarScalarFactor_smul_congr' ..
"""
        self.assertFalse(_kept(chosen, rejected))

    def test_renamed_lemma_in_the_same_script_is_not_kept(self) -> None:
        rejected = (
            "by rw [clog, if_pos] exact Nat.pos_of_ne_zero (clog.go_spec hn hb).1"
        )
        chosen = "by rw [clog, if_pos] exact Nat.succ_pos _"
        self.assertEqual(proof_diff_class(chosen, rejected), "minor")

    def test_api_edits_are_not_kept(self) -> None:
        pairs = [
            ("obtain rfl | hc := eq_or_ne c 0", "obtain rfl | hc := eq_zero_or_pos c"),
            (
                "exact (nthRoots n 1).toFinset.finite_toSet",
                "exact (nthRoots (↑n) 1).toFinset.finite_toSet",
            ),
            (
                "have IH (n : ℕ) : (P.eval t - t).sign = (P.eval t - t).sign := by rfl",
                "have IH : ∀ n : ℕ, (P.eval t - t).sign = (P.eval t - t).sign := fun n ↦ by rfl",
            ),
            (
                "preservesColimit_of_natIso F (BraidedCategory.tensorLeftIsoTensorRight c).symm",
                "preservesColimit_of_natIso F (NatIso.ofComponents (fun _ ↦ β_ _ _))",
            ),
            (
                "· rintro p k hp - -\n  simp [hf]\n· rintro q hq\n  simp [hq]",
                "· rintro q hq\n  simp [hq]\n· rintro p k hp - -\n  simp [hf]",
            ),
        ]
        for chosen, rejected in pairs:
            self.assertNotIn(
                proof_diff_class(chosen, rejected),
                KEEP_CLASSES,
                msg=chosen,
            )

    def test_simp_script_versus_ext_rw_is_kept(self) -> None:
        rejected = "simp only [foo, bar, baz]"
        chosen = "ext; rw [qux]"
        self.assertEqual(proof_diff_class(chosen, rejected), "construction")


class FilterCommandTests(unittest.TestCase):
    def test_writes_kept_rows_and_replaces_the_readme_section(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "pairs.jsonl"
            rows = [
                {
                    "prompt": "p",
                    "chosen": "cases x <;> simp",
                    "rejected": "by grind",
                    "metadata": {"declaration_name": "keepMe"},
                },
                {
                    "prompt": "p",
                    "chosen": "exact (foo)",
                    "rejected": "exact foo",
                    "metadata": {"declaration_name": "dropMe"},
                },
            ]
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            readme = root / "README.md"
            readme.write_text("# Pairs\n\nExisting note.\n", encoding="utf-8")
            output = root / "pairs.substantive.jsonl"
            annotated = root / "pairs.annotated.jsonl"
            argv = [
                "--input",
                str(source),
                "--output",
                str(output),
                "--annotated",
                str(annotated),
                "--readme",
                str(readme),
            ]
            with redirect_stdout(StringIO()):
                main(argv)
                main(argv)
            kept = [json.loads(line) for line in output.read_text().splitlines()]
            labeled = [json.loads(line) for line in annotated.read_text().splitlines()]
            self.assertEqual(
                [row["metadata"]["declaration_name"] for row in kept],
                ["keepMe"],
            )
            self.assertEqual(kept[0]["metadata"]["diff_class"], "construction")
            self.assertEqual(
                [row["metadata"]["diff_class"] for row in labeled],
                ["construction", "layout"],
            )
            text = readme.read_text(encoding="utf-8")
            self.assertEqual(text.count("## Filtering convention edits"), 1)
            self.assertIn("Existing note.", text)
            self.assertIn("(1 of 2)", text)
            self.assertLess(
                text.index("## Filtering convention edits"),
                text.index("(1 of 2)"),
            )

    def test_filter_section_stays_before_the_length_note(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "pairs.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "prompt": "p",
                        "chosen": "exact (foo)",
                        "rejected": "exact foo",
                        "metadata": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            readme = root / "README.md"
            readme.write_text(
                "# Pairs\n\n### On length bias\n\nShorter is common.\n\n"
                "## Filtering convention edits\n\nOld counts.\n",
                encoding="utf-8",
            )
            argv = [
                "--input",
                str(source),
                "--output",
                str(root / "kept.jsonl"),
                "--annotated",
                str(root / "annotated.jsonl"),
                "--readme",
                str(readme),
            ]
            with redirect_stdout(StringIO()):
                main(argv)
            text = readme.read_text(encoding="utf-8")
            self.assertEqual(text.count("## Filtering convention edits"), 1)
            self.assertEqual(text.count("### On length bias"), 1)
            self.assertLess(
                text.index("## Filtering convention edits"),
                text.index("### On length bias"),
            )
            self.assertIn("Shorter is common.", text)
            self.assertNotIn("Old counts.", text)


if __name__ == "__main__":
    unittest.main()
