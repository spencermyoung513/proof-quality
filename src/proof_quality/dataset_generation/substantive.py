"""Classify proof edits as Mathlib convention or a real construction change.

Convention edits are renames, layout, tactic synonyms, and simp-set churn.
A construction change is a different proof script: a new structural step, a
hammer traded for a written proof, or a call pattern that is not a rename.
Comment-only edits are kept when the comment wording itself changes.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from proof_quality.dataset_generation.lean_source import (
    normalize_whitespace,
    split_comments,
)


KEEP_CLASSES = frozenset({"comment", "construction"})

# Closing and bookkeeping tactics that Mathlib treats as interchangeable
# when the argument is the same proof term or the same rewrite.
_SYNONYMS = {
    "apply": "exact",
    "refine": "exact",
    "simpa": "simp",
    "dsimp": "simp",
    "simp_all": "simp",
    "field_simp": "simp",
    "erw": "rw",
    "simp_rw": "rw",
    "haveI": "have",
    "letI": "let",
    "funext": "ext",
    "rcases": "cases",
    "cases'": "cases",
    "obtain": "cases",
    "rintro": "intro",
    "intros": "intro",
    "induction'": "induction",
    "cat_disch": "aesop",
}

_STRUCTURAL = frozenset(
    {
        "have",
        "let",
        "calc",
        "induction",
        "cases",
        "suffices",
        "conv",
        "by_contra",
        "by_cases",
        "split",
        "split_ifs",
        "intro",
        "show",
        "change",
        "convert",
        "ext",
    }
)

_HAMMERS = frozenset(
    {
        "grind",
        "aesop",
        "tauto",
        "omega",
        "linarith",
        "nlinarith",
        "ring",
        "ring_nf",
        "decide",
        "native_decide",
        "positivity",
        "norm_num",
        "abel",
    }
)

_TACTICS = frozenset(
    _STRUCTURAL
    | _HAMMERS
    | set(_SYNONYMS)
    | {
        "exact",
        "simp",
        "rw",
        "constructor",
        "trivial",
        "rfl",
        "assumption",
        "contradiction",
        "exfalso",
        "left",
        "right",
        "use",
        "specialize",
        "unfold",
        "all_goals",
        "any_goals",
        "exact_mod_cast",
        "norm_cast",
        "push_cast",
        "revert",
        "generalize",
        "subst",
        "injection",
        "classical",
        "repeat",
        "congr",
        "gcongr",
        "apply_fun",
        "nth_rw",
        "set",
        "clear",
        "borelize",
        "fun_prop",
    }
)

_LAYOUT = frozenset({"·", "<;>", ";"})
_PUNCT = frozenset("()[]{}⟨⟩⦃⦄,.:;|") | {"=>", "↦", "·", "<;>", ";"}
_OPEN = {"(": ")", "[": "]", "{": "}", "⟨": "⟩", "⦃": "⦄"}
_CLOSE = {closer: opener for opener, closer in _OPEN.items()}

_TOKEN = re.compile(
    r"<;>|\|>\.|\|>|\|<|<\|"
    r"|=>|↦|←|→|↔"
    r"|[()\[\]{}⟨⟩⦃⦄,.;]"
    r"|@?[^\W\d][\w'.]*"
    r"|[^\s]",
    flags=re.UNICODE,
)


def proof_diff_class(chosen: str, rejected: str) -> str:
    """Return how ``chosen`` differs from ``rejected``.

    Kept classes are :data:`KEEP_CLASSES`. Every other class is a convention
    edit: comment spacing, tactic style, punctuation, a rename, a simp or
    ``rw`` list edit, or a non-structural tweak.
    """
    chosen_code, chosen_comments = split_comments(chosen)
    rejected_code, rejected_comments = split_comments(rejected)
    if normalize_whitespace(chosen_code) == normalize_whitespace(rejected_code):
        if _comment_key(chosen_comments) != _comment_key(rejected_comments):
            return "comment"
        return "comment_spacing"

    chosen_tokens = _normalize_code(chosen_code)
    rejected_tokens = _normalize_code(rejected_code)
    if chosen_tokens == rejected_tokens or _same_after_inlining(
        rejected_tokens, chosen_tokens
    ):
        return "style"
    if _punctuation_only(rejected_tokens, chosen_tokens):
        return "layout"
    if _is_rename(rejected_tokens, chosen_tokens):
        return "rename"
    if _is_trans_chain(rejected_tokens, chosen_tokens):
        return "style"
    if _is_rewrite_list(rejected_tokens, chosen_tokens):
        return "simp_list"
    if _is_bookkeeping(rejected_tokens, chosen_tokens):
        return "bookkeeping"
    if _is_construction(rejected_tokens, chosen_tokens):
        return "construction"
    return "minor"


def _comment_key(comments: str) -> str:
    return re.sub(r"\s+", "", comments)


def _normalize_code(code: str) -> list[str]:
    tokens = [_strip_at(token) for token in _TOKEN.findall(code)]
    tokens = _desugar_pipes(tokens)
    tokens = _desugar_rev_app(tokens)
    tokens = [token for token in tokens if token not in _LAYOUT]
    tokens = [_SYNONYMS.get(token, token) for token in tokens]
    tokens = _peel_term_wrapper(tokens)
    tokens = _canonicalize_card(tokens)
    return _canonicalize_borelize(tokens)


def _strip_at(token: str) -> str:
    if token.startswith("@") and len(token) > 1 and _is_ident(token[1:]):
        return token[1:]
    return token


def _is_ident(token: str) -> bool:
    return bool(re.fullmatch(r"@?[^\W\d][\w'.]*", token, flags=re.UNICODE))


def _is_term_atom(token: str) -> bool:
    if token in _TACTICS or token in _SYNONYMS:
        return False
    if _is_ident(token) or token == "_":
        return True
    return token[:1].isdigit()


def _peel_term_wrapper(tokens: list[str]) -> list[str]:
    """Drop a leading ``by exact`` when the whole proof is one term."""
    if tokens and tokens[0] == "by":
        tokens = tokens[1:]
    if (
        tokens
        and tokens[0] == "exact"
        and not any(token in _TACTICS for token in tokens[1:])
    ):
        return tokens[1:]
    return tokens


def _is_content_ident(token: str) -> bool:
    return _is_ident(token) and token not in _TACTICS and token != "_"


def _canonicalize_card(tokens: list[str]) -> list[str]:
    """Rewrite ``#s`` to ``s.card``. Both spell ``Finset.card``."""
    rewritten: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if (
            token == "#"
            and index + 1 < len(tokens)
            and _is_content_ident(tokens[index + 1])
        ):
            name = tokens[index + 1]
            rewritten.append(name if name.endswith(".card") else f"{name}.card")
            index += 2
            continue
        rewritten.append(token)
        index += 1
    return rewritten


def _canonicalize_borelize(tokens: list[str]) -> list[str]:
    """Rewrite the ``obtain``/``let`` borel boilerplate as ``borelize``."""
    pattern = [
        "cases",
        "⟨",
        "rfl",
        "⟩",
        ":",
        "=",
        "‹",
        "BorelSpace",
        None,
        "›",
        "let",
        ":",
        "=",
        "borel",
        None,
    ]
    rewritten: list[str] = []
    index = 0
    while index < len(tokens):
        matched = _match_pattern(tokens, index, pattern)
        if matched is None:
            rewritten.append(tokens[index])
            index += 1
            continue
        type_name, end = matched
        rewritten.extend(["borelize", type_name])
        index = end
    return rewritten


def _match_pattern(
    tokens: list[str], start: int, pattern: list[str | None]
) -> tuple[str, int] | None:
    """Match ``pattern``. ``None`` is a wildcard ident, and both must be the same."""
    if start + len(pattern) > len(tokens):
        return None
    captured: str | None = None
    for offset, expected in enumerate(pattern):
        token = tokens[start + offset]
        if expected is None:
            if not _is_content_ident(token):
                return None
            if captured is None:
                captured = token
            elif token != captured:
                return None
            continue
        if token != expected:
            return None
    if captured is None:
        return None
    return captured, start + len(pattern)


def _inline_trivial_haves(tokens: list[str]) -> list[str]:
    """Splice a term-mode ``have`` into the later ``simp`` or ``rw`` that names it."""
    changed = True
    while changed:
        changed = False
        index = 0
        while index < len(tokens):
            parsed = _parse_term_have(tokens, index)
            if parsed is None:
                index += 1
                continue
            name, term, body_end = parsed
            uses = [
                pos
                for pos in range(body_end, len(tokens))
                if tokens[pos] == name and _inside_brackets(tokens, pos)
            ]
            if len(uses) != 1:
                index += 1
                continue
            use = uses[0]
            tokens = tokens[:index] + tokens[body_end:use] + term + tokens[use + 1 :]
            changed = True
            break
    return tokens


def _parse_term_have(
    tokens: list[str], start: int, keyword: str = "have"
) -> tuple[str, list[str], int] | None:
    """Parse ``have [name] [: type] := <term>`` with no tactics in the term."""
    if start >= len(tokens) or tokens[start] != keyword:
        return None
    index = start + 1
    name = "this"
    if index < len(tokens) and _is_content_ident(tokens[index]):
        name = tokens[index]
        index += 1
    if index < len(tokens) and tokens[index] == ":" and not _is_coloneq(tokens, index):
        index += 1
        depth = 0
        while index < len(tokens):
            token = tokens[index]
            if token in _OPEN:
                depth += 1
            elif token in _CLOSE and depth:
                depth -= 1
            elif depth == 0 and _is_coloneq(tokens, index):
                break
            elif depth == 0 and token in _TACTICS:
                return None
            index += 1
    if not _is_coloneq(tokens, index):
        return None
    index += 2
    term_start = index
    depth = 0
    while index < len(tokens):
        token = tokens[index]
        if token in _OPEN:
            depth += 1
        elif token in _CLOSE and depth:
            depth -= 1
        elif depth == 0 and token == "by":
            return None
        elif depth == 0 and token in _TACTICS:
            break
        index += 1
    term = tokens[term_start:index]
    if not term or any(token in _TACTICS for token in term):
        return None
    return name, term, index


def _is_coloneq(tokens: list[str], index: int) -> bool:
    return index + 1 < len(tokens) and tokens[index] == ":" and tokens[index + 1] == "="


def _inside_brackets(tokens: list[str], index: int) -> bool:
    depth = 0
    for cursor in range(index):
        if tokens[cursor] == "[":
            depth += 1
        elif tokens[cursor] == "]" and depth:
            depth -= 1
    return depth > 0


def _merge_rw_into_simp(tokens: list[str]) -> list[str]:
    """Fold ``rw [lemmas]; simp [more]`` into one ``simp`` list."""
    rewritten: list[str] = []
    index = 0
    while index < len(tokens):
        if tokens[index] != "rw":
            rewritten.append(tokens[index])
            index += 1
            continue
        rw_list = _rewrite_bracket(tokens, index + 1)
        if rw_list is None:
            rewritten.append(tokens[index])
            index += 1
            continue
        rw_items, after_rw = rw_list
        if after_rw >= len(tokens) or tokens[after_rw] != "simp":
            rewritten.append(tokens[index])
            index += 1
            continue
        simp_list = _rewrite_bracket(tokens, after_rw + 1)
        if simp_list is None:
            rewritten.append(tokens[index])
            index += 1
            continue
        simp_items, after_simp = simp_list
        combined = _join_list_items(rw_items, simp_items)
        rewritten.extend(["simp", "["])
        rewritten.extend(combined)
        rewritten.append("]")
        index = after_simp
    return rewritten


def _join_list_items(left: list[str], right: list[str]) -> list[str]:
    if left and right:
        return left + [","] + right
    return left + right


def _rewrite_bracket(tokens: list[str], index: int) -> tuple[list[str], int] | None:
    """Return the inside of ``[only] [...]`` and the index just after it."""
    if index < len(tokens) and tokens[index] == "!":
        index += 1
    if index < len(tokens) and tokens[index] == "only":
        index += 1
    if index >= len(tokens) or tokens[index] != "[":
        return None
    close = _group_close(tokens, index)
    return tokens[index + 1 : close], close + 1


def _desugar_pipes(tokens: list[str]) -> list[str]:
    """Rewrite ``receiver |>.name`` as ``name (receiver)``."""
    tokens = list(tokens)
    while True:
        found = False
        for index, token in enumerate(tokens):
            if token not in {"|>.", "|>"}:
                continue
            if index + 1 >= len(tokens) or not _is_ident(tokens[index + 1]):
                continue
            start = _term_start(tokens, index)
            if start >= index:
                continue
            name = tokens[index + 1]
            receiver = tokens[start:index]
            tokens = (
                tokens[:start] + [name, "("] + receiver + [")"] + tokens[index + 2 :]
            )
            found = True
            break
        if not found:
            return tokens


def _desugar_rev_app(tokens: list[str]) -> list[str]:
    """Rewrite right-associative ``f <| x`` as ``f (x)``."""
    tokens = list(tokens)
    while True:
        indexes = [index for index, token in enumerate(tokens) if token == "<|"]
        if not indexes:
            return tokens
        index = indexes[-1]
        left_start = _term_start(tokens, index)
        right_end = _term_end(tokens, index + 1)
        if left_start >= index or right_end <= index + 1:
            tokens = tokens[:index] + tokens[index + 1 :]
            continue
        left = tokens[left_start:index]
        right = tokens[index + 1 : right_end]
        tokens = tokens[:left_start] + left + ["("] + right + [")"] + tokens[right_end:]


def _term_start(tokens: list[str], end: int) -> int:
    """Index where the term ending at ``end`` begins."""
    index = end
    while index > 0:
        token = tokens[index - 1]
        if token in _CLOSE:
            index = _group_open(tokens, index - 1)
            continue
        if _is_term_atom(token):
            index -= 1
            continue
        break
    return index


def _term_end(tokens: list[str], start: int) -> int:
    """Index just after the term beginning at ``start``."""
    index = start
    if index >= len(tokens):
        return index
    if tokens[index] not in _OPEN and not _is_term_atom(tokens[index]):
        return index
    while index < len(tokens):
        token = tokens[index]
        if token in _OPEN:
            index = _group_close(tokens, index) + 1
            continue
        if _is_term_atom(token):
            index += 1
            continue
        break
    return index


def _group_open(tokens: list[str], close_index: int) -> int:
    closer = tokens[close_index]
    opener = _CLOSE[closer]
    depth = 1
    index = close_index - 1
    while index >= 0 and depth:
        if tokens[index] == closer:
            depth += 1
        elif tokens[index] == opener:
            depth -= 1
        index -= 1
    return index + 1


def _group_close(tokens: list[str], open_index: int) -> int:
    opener = tokens[open_index]
    closer = _OPEN[opener]
    depth = 1
    index = open_index + 1
    while index < len(tokens) and depth:
        if tokens[index] == opener:
            depth += 1
        elif tokens[index] == closer:
            depth -= 1
        index += 1
    return index - 1


def _opcodes(rejected: list[str], chosen: list[str]):
    matcher = SequenceMatcher(a=rejected, b=chosen, autojunk=False)
    return [op for op in matcher.get_opcodes() if op[0] != "equal"]


def _punctuation_only(rejected: list[str], chosen: list[str]) -> bool:
    for _tag, i1, i2, j1, j2 in _opcodes(rejected, chosen):
        if any(token not in _PUNCT for token in rejected[i1:i2] + chosen[j1:j2]):
            return False
    return True


def _is_rename(rejected: list[str], chosen: list[str]) -> bool:
    """True when every edit replaces identifiers in place.

    A replaced name counts even when the strings do not look alike. Inserting
    or deleting a token is a different call pattern, not a rename.
    """
    ops = _opcodes(rejected, chosen)
    if not ops:
        return False
    for tag, i1, i2, j1, j2 in ops:
        if tag != "replace" or (i2 - i1) != (j2 - j1):
            return False
        pairs = zip(rejected[i1:i2], chosen[j1:j2], strict=True)
        if not all(_is_ident(left) and _is_ident(right) for left, right in pairs):
            return False
    return True


_TRANS_COMBINATOR = re.compile(
    r"^(?:(?:lt|le|ge|gt)_of_)*(?:trans|lt|le|ge|gt)(?:_(?:trans|lt|le|ge|gt))*$"
)


def _is_trans_combinator(token: str) -> bool:
    """True for ``trans``, ``le_trans``, ``lt_of_le_of_lt``, ``trans_lt``, and the like."""
    return bool(_TRANS_COMBINATOR.fullmatch(token.split(".")[-1]))


def _is_trans_chain(rejected: list[str], chosen: list[str]) -> bool:
    """True when the edit only reassociates a transitivity chain.

    ``lt_of_le_of_lt (le_trans a b) c`` and ``a |>.trans <| b |>.trans_lt <| c``
    are the same argument written in Mathlib's pipe style.
    """
    for _tag, i1, i2, j1, j2 in _opcodes(rejected, chosen):
        changed = rejected[i1:i2] + chosen[j1:j2]
        if any(
            token not in _PUNCT and not _is_trans_combinator(token) for token in changed
        ):
            return False
    return True


def _same_after_inlining(rejected: list[str], chosen: list[str]) -> bool:
    """True when both sides match once a term-mode ``have`` is folded into ``simp``."""
    return _merge_rw_into_simp(_inline_trivial_haves(rejected)) == _merge_rw_into_simp(
        _inline_trivial_haves(chosen)
    )


def _is_bookkeeping(rejected: list[str], chosen: list[str]) -> bool:
    """True when the edit only adds binders, holes, projections, or explicit arguments.

    Replacing one lemma name with a different lemma name is not bookkeeping.
    """
    operations = _opcodes(rejected, chosen)
    if not operations:
        return False
    for tag, i1, i2, j1, j2 in operations:
        left = rejected[i1:i2]
        right = chosen[j1:j2]
        if tag == "replace" and _swaps_lemma(left, right):
            return False
        if any(not _is_book_token(token) for token in left + right):
            return False
    return True


def _swaps_lemma(left: list[str], right: list[str]) -> bool:
    left_names = [token for token in left if _is_content_ident(token)]
    right_names = [token for token in right if _is_content_ident(token)]
    return bool(left_names) and bool(right_names) and left_names != right_names


def _is_book_token(token: str) -> bool:
    if token in _PUNCT or token in {"?", "by", "#", ".", "_"} or token.isdigit():
        return True
    return _is_content_ident(token)


def _is_rewrite_list(rejected: list[str], chosen: list[str]) -> bool:
    """True when the scripts match once ``simp`` and ``rw`` arguments are ignored."""
    stripped_rejected = _strip_rewrite_lists(rejected)
    stripped_chosen = _strip_rewrite_lists(chosen)
    return stripped_rejected == stripped_chosen


def _strip_rewrite_lists(tokens: list[str]) -> list[str]:
    kept: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        kept.append(token)
        index += 1
        if token not in {"simp", "rw"}:
            continue
        if index < len(tokens) and tokens[index] == "!":
            index += 1
        if index < len(tokens) and tokens[index] == "only":
            index += 1
        if index < len(tokens) and tokens[index] == "[":
            index = _group_close(tokens, index) + 1
    return kept


def _heads(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token in _TACTICS]


def _is_construction(rejected: list[str], chosen: list[str]) -> bool:
    """True when the script gains or loses a real proof step or a hammer.

    The same tactics applied to a renamed lemma, a coercion, or a reordered
    bullet are not a different construction. A ``have`` or ``let`` counts only
    when its body is itself a proof script, not a term or an instance.
    """
    if _script_structure(rejected) != _script_structure(chosen):
        return True
    rejected_hammers = set(_heads(rejected)) & _HAMMERS
    chosen_hammers = set(_heads(chosen)) & _HAMMERS
    return rejected_hammers != chosen_hammers


_TERM_BINDERS = frozenset({"have", "let"})


def _script_structure(tokens: list[str]) -> frozenset[str]:
    present = set(_heads(tokens)) & _STRUCTURAL
    hard = present - _TERM_BINDERS
    for keyword in present & _TERM_BINDERS:
        if _binder_has_script(tokens, keyword):
            hard.add(keyword)
    return frozenset(hard)


def _binder_has_script(tokens: list[str], keyword: str) -> bool:
    index = 0
    while index < len(tokens):
        if tokens[index] != keyword:
            index += 1
            continue
        parsed = _parse_term_have(tokens, index, keyword)
        if parsed is None:
            return True
        index = parsed[2]
    return False
