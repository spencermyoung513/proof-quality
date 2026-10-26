import unittest

from proof_quality.dataset_generation.lean_source import is_sorry_proof, parse_lean_file


SAMPLE = """
variable {E : Type*} [Group E]
variable {v : E} {unused : Nat}

/-- doc -/
theorem foo (h : v = v) : True := by
  trivial

theorem other (n : Nat) : n = n := by
  rfl

namespace Bar

lemma baz : True := by
  trivial

end Bar
"""


class ParseLeanFileTest(unittest.TestCase):
    def test_inlines_transitive_binders_and_skips_unused(self) -> None:
        decls = {decl.name: decl for decl in parse_lean_file(SAMPLE)}
        signature, text = decls["foo"].rendered()
        self.assertIn("{E : Type*}", signature)
        self.assertIn("[Group E]", signature)
        self.assertIn("{v : E}", signature)
        self.assertNotIn("unused", signature)
        self.assertNotIn("/--", text)
        self.assertIn("trivial", text)
        other_sig, _ = decls["other"].rendered()
        self.assertNotIn("{v : E}", other_sig)
        self.assertEqual(decls["baz"].qualified_name, "Bar.baz")

    def test_section_end_drops_binders(self) -> None:
        text = """
section
variable (x : Nat)
theorem inside : x = x := by
  rfl
end

theorem outside : True := by
  trivial
"""
        decls = {decl.name: decl for decl in parse_lean_file(text)}
        self.assertIn("(x : Nat)", decls["inside"].rendered()[0])
        self.assertNotIn("(x : Nat)", decls["outside"].rendered()[0])

    def test_sorry_and_define_inside_comment(self) -> None:
        text = """
theorem commented : True := by
  -- := sorry
  trivial

theorem hole : True := by
  sorry
"""
        decls = {decl.name: decl for decl in parse_lean_file(text)}
        self.assertFalse(is_sorry_proof(decls["commented"].body))
        self.assertTrue(is_sorry_proof(decls["hole"].body))
        self.assertIn("trivial", decls["commented"].body)

    def test_explicit_rebind_is_not_copied_twice(self) -> None:
        text = """
variable {s : Set Nat} {x : Nat} {L : Filter Nat}
variable (s x L)

theorem foo : x ∈ s := by
  rfl
"""
        signature, _ = parse_lean_file(text)[0].rendered()
        self.assertIn("{s : Set Nat}", signature)
        self.assertIn("{x : Nat}", signature)
        self.assertNotIn("(s x L)", signature)

    def test_variable_in_applies_only_to_the_next_command(self) -> None:
        text = """
variable {f : Nat}

variable (f x) in
def ignored : Nat := f

variable (x : Nat) in
/-- only the next theorem -/
theorem onlyNext : x = x := by
  rfl

theorem later (y : Nat) : y = y := by
  rfl
"""
        decls = {decl.name: decl for decl in parse_lean_file(text)}
        only_sig, _ = decls["onlyNext"].rendered()
        later_sig, _ = decls["later"].rendered()
        self.assertIn("(x : Nat)", only_sig)
        self.assertNotIn("(f x)", only_sig)
        self.assertNotIn("(x : Nat)", later_sig)
        self.assertNotIn("(f x)", later_sig)

    def test_universe_levels_stay_out_of_the_name(self) -> None:
        decls = parse_lean_file("theorem foo.{u} {α : Type u} : True := trivial\n")
        self.assertEqual(decls[0].name, "foo")
        self.assertIn(".{u}", decls[0].rendered()[0])


if __name__ == "__main__":
    unittest.main()
