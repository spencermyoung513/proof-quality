"""Source-level Lean 4 declaration parsing.

Snapshots in MathlibPR can sit on different Lean toolchains, so declarations are
recovered from the file text rather than from an elaborator.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


_MODIFIERS = (
    "private",
    "protected",
    "noncomputable",
    "unsafe",
    "partial",
    "scoped",
    "local",
)
_OPENERS = {"(": ")", "[": "]", "{": "}", "⦃": "⦄"}
_CLOSERS = {closer: opener for opener, closer in _OPENERS.items()}
_CONTINUATION_PREFIXES = ("where", "termination_by", "decreasing_by")


@dataclass(frozen=True)
class Declaration:
    """One top-level theorem or lemma, with the binders in scope around it."""

    kind: str
    name: str
    qualified_name: str
    start_line: int
    end_line: int
    modifiers: str
    attributes: str
    rest: str
    body: str
    binder_groups: tuple[str, ...]

    def rendered(self) -> tuple[str, str]:
        """Return `(signature, full_text)` with in-scope binders copied in.

        `signature` stops before `:=`. `full_text` is the standalone declaration
        and its proof, without the docstring.
        """
        groups = _select_binder_groups(self.binder_groups, self.rest, self.body)
        groups = [group for group in groups if not _group_covered(self.rest, group)]
        groups = _drop_rebinders(groups)
        signature = _join_signature(
            self.modifiers, self.kind, self.name, groups, self.rest
        )
        proof = self.body
        if proof.startswith(("\n", " ")):
            text = f"{signature} :={proof}"
        elif proof:
            text = f"{signature} := {proof.lstrip()}"
        else:
            text = f"{signature} :="
        if self.attributes:
            text = f"{self.attributes}\n{text}"
        return signature, text


def proof_text(body: str) -> str:
    """Return the proof that follows `:=`, without a copied declaration.

    A body copied from `:= by` starts with one space. A body whose proof is
    on the next line starts with a newline. Drop that separator and keep the
    proof's own indentation.
    """
    if body.startswith(" "):
        body = body[1:]
    elif body.startswith("\n"):
        body = body[1:]
    return body.rstrip()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def comparable_proof(body: str) -> str:
    """Proof text for equality. Comments stay, and whitespace is collapsed.

    A comment-only edit is a real difference. A space-only or line-break-only
    edit is not.
    """
    return normalize_whitespace(body)


def normalized_proof(body: str) -> str:
    """Proof text with comments removed and whitespace collapsed."""
    return normalize_whitespace(_strip_comments(body))


def is_sorry_proof(body: str) -> bool:
    return normalized_proof(body) in {"sorry", "by sorry"}


def parse_lean_file(text: str) -> list[Declaration]:
    """Parse top-level theorem and lemma declarations from Lean source."""
    return _parse_lines(text.splitlines())


def _parse_lines(lines: list[str]) -> list[Declaration]:
    frames: list[_Frame] = [_Frame("root", None)]
    declarations: list[Declaration] = []
    index = 0
    pending_doc: tuple[int, int] | None = None
    pending_attrs: tuple[int, int] | None = None
    # `variable (x) in` binds `x` only on the next command, not the rest of the section.
    pending_local: list[str] = []
    while index < len(lines):
        line = lines[index]
        if not line.strip() or line.startswith((" ", "\t")):
            if not line.strip():
                pending_doc = None
                pending_attrs = None
            index += 1
            continue
        stripped = line.strip()
        if stripped.startswith("--"):
            pending_doc = None
            pending_attrs = None
            index += 1
            continue
        if stripped.startswith("/-"):
            start = index + 1
            _, index = _consume_comment_block(lines, index)
            if stripped.startswith(("/--", "/-!")):
                pending_doc = (start, index)
            else:
                pending_doc = None
                pending_attrs = None
            continue
        if stripped.startswith("@["):
            start = index + 1
            _, index = _consume_balanced_lines(lines, index)
            pending_attrs = (start, index)
            continue
        if _declaration_start(stripped):
            doc_start = pending_doc[0] if pending_doc else index + 1
            attributes = ""
            if pending_attrs:
                attributes = "\n".join(lines[pending_attrs[0] - 1 : pending_attrs[1]])
            decl, index = _read_declaration(
                lines,
                index,
                frames,
                doc_start=doc_start,
                attributes=attributes,
                local_groups=pending_local,
            )
            pending_doc = None
            pending_attrs = None
            pending_local = []
            if decl is not None:
                declarations.append(decl)
            continue
        pending_doc = None
        pending_attrs = None
        if stripped.startswith("variable"):
            command_start = index
            _, index = _consume_balanced_lines(lines, index)
            while (
                index < len(lines)
                and lines[index].startswith((" ", "\t"))
                and lines[index].strip()
            ):
                _, index = _consume_balanced_lines(lines, index)
            command = "\n".join(lines[command_start:index]).strip()
            local, groups = _variable_command(command)
            if local:
                pending_local = groups
            else:
                frames[-1].variables.extend(groups)
            continue
        pending_local = []
        command_start = index
        _, index = _consume_balanced_lines(lines, index)
        command = "\n".join(lines[command_start:index]).strip()
        _apply_command(frames, command)
    return declarations


@dataclass
class _Frame:
    kind: str
    name: str | None
    variables: list[str]
    omitted: set[str]

    def __init__(self, kind: str, name: str | None):
        self.kind = kind
        self.name = name
        self.variables = []
        self.omitted = set()


def _declaration_start(stripped: str) -> bool:
    words = stripped.split()
    if not words:
        return False
    index = 0
    while index < len(words) and words[index] in _MODIFIERS:
        index += 1
    return index < len(words) and words[index] in {"theorem", "lemma"}


def _read_declaration(
    lines: list[str],
    keyword_index: int,
    frames: list[_Frame],
    doc_start: int,
    attributes: str,
    local_groups: list[str],
) -> tuple[Declaration | None, int]:
    end = keyword_index + 1
    while end < len(lines):
        line = lines[end]
        if not line.strip() or line.startswith((" ", "\t")):
            end += 1
            continue
        if line.strip().startswith(_CONTINUATION_PREFIXES):
            end += 1
            continue
        break
    while end > keyword_index + 1 and not lines[end - 1].strip():
        end -= 1
    source = "\n".join(lines[keyword_index:end])
    parsed = _split_declaration(source)
    if parsed is None:
        return None, end
    modifiers, kind, name, rest, body = parsed
    namespace = _namespace(frames)
    if name.startswith("_root_."):
        qualified = name[len("_root_.") :]
    elif namespace:
        qualified = f"{namespace}.{name}"
    else:
        qualified = name
    declaration = Declaration(
        kind=kind,
        name=name,
        qualified_name=qualified,
        start_line=doc_start,
        end_line=end,
        modifiers=modifiers,
        attributes=attributes.strip(),
        rest=rest,
        body=body,
        binder_groups=_binders_in_scope(frames, local_groups),
    )
    return declaration, end


def _split_declaration(
    source: str,
) -> tuple[str, str, str, str, str] | None:
    header, separator, tail = _split_define(source)
    if separator is None or header is None:
        return None
    match = _HEADER_RE.match(header.lstrip())
    if match is None:
        return None
    modifiers = " ".join(match.group("mods").split())
    kind = match.group("kind")
    name = match.group("name")
    rest = header.lstrip()[match.end() :]
    return modifiers, kind, name, rest, tail


def _compile_header() -> re.Pattern[str]:
    # `\w` covers Lean identifiers such as `α` and `f₁`. Dots stay outside a
    # single ident so `Foo.bar` is a qualified name and `foo.{u}` is not.
    ident = r"(?:«[^»]+»|[^\W\d_][\w']*|_[^\W\d][\w']*)"
    return re.compile(
        r"^(?P<mods>(?:(?:private|protected|noncomputable|unsafe|partial|scoped|local)\s+)*)"
        rf"(?P<kind>theorem|lemma)\s+(?P<name>_root_\.{ident}(?:\.{ident})*|{ident}(?:\.{ident})*)",
        re.S,
    )


_HEADER_RE = _compile_header()


def _split_define(source: str) -> tuple[str | None, str | None, str]:
    """Split on the definitional `:=` that is outside comments, strings, and groups."""
    index = 0
    depth = 0
    length = len(source)
    while index < length:
        if source.startswith("--", index):
            newline = source.find("\n", index)
            index = length if newline < 0 else newline
            continue
        if source.startswith("/-", index):
            index = _skip_block_comment(source, index)
            continue
        char = source[index]
        if char == '"':
            index = _skip_string(source, index)
            continue
        if char in _OPENERS:
            depth += 1
        elif char in _CLOSERS and depth:
            depth -= 1
        elif depth == 0 and source.startswith(":=", index):
            return source[:index], ":=", source[index + 2 :]
        index += 1
    return None, None, ""


def _select_binder_groups(groups: tuple[str, ...], rest: str, body: str) -> list[str]:
    """Keep section binders needed for the declaration to mention the same names."""
    bound = _bound_names(rest)
    selected: list[str] = []
    text_idents = _identifier_set(f"{rest}\n{body}") - bound
    changed = True
    while changed:
        changed = False
        for group in groups:
            if group in selected:
                continue
            names = _binder_names(group)
            if names:
                if any(name in text_idents for name in names):
                    selected.append(group)
                    text_idents |= _identifier_set(group)
                    changed = True
                continue
            if group.startswith("[") and _identifier_set(group) & text_idents:
                selected.append(group)
                text_idents |= _identifier_set(group)
                changed = True
    return [group for group in groups if group in selected]


def _drop_rebinders(groups: list[str]) -> list[str]:
    """Drop a later group whose names were already introduced by an earlier one.

    Mathlib writes `variable {s : Set α} (s)` to make `s` explicit. Copying both
    groups onto one declaration would bind `s` twice.
    """
    bound: set[str] = set()
    kept: list[str] = []
    for group in groups:
        names = _binder_names(group)
        if names and all(name in bound for name in names):
            continue
        kept.append(group)
        bound.update(names)
    return kept


def _group_covered(rest: str, group: str) -> bool:
    names = _binder_names(group)
    if names:
        bound = _bound_names(rest)
        return all(name in bound for name in names)
    return normalize_whitespace(group) in normalize_whitespace(rest)


def _join_signature(
    modifiers: str, kind: str, name: str, groups: list[str], rest: str
) -> str:
    prefix = f"{modifiers} {kind}".strip()
    head = f"{prefix} {name}"
    if not groups:
        return f"{head}{rest}".rstrip()
    inserted = "\n    ".join(groups)
    if rest.startswith("\n"):
        return f"{head}\n    {inserted}{rest}".rstrip()
    spacer = "" if not rest or rest.startswith((" ", "\n")) else " "
    return f"{head}\n    {inserted}{spacer}{rest}".rstrip()


def _binders_in_scope(frames: list[_Frame], local_groups: list[str]) -> tuple[str, ...]:
    omitted: set[str] = set()
    for frame in frames:
        omitted.update(frame.omitted)
    groups: list[str] = []
    for frame in frames:
        groups.extend(frame.variables)
    groups.extend(local_groups)
    seen: set[str] = set()
    unique: list[str] = []
    for group in groups:
        names = _binder_names(group)
        if names and all(name in omitted for name in names):
            continue
        key = normalize_whitespace(group)
        if key in seen:
            continue
        seen.add(key)
        unique.append(group)
    return tuple(unique)


def _variable_command(command: str) -> tuple[bool, list[str]]:
    """Return `(local_to_next_command, binder_groups)` for a `variable` command."""
    body = command.strip()[len("variable") :].strip()
    body = re.sub(r"\s*--.*$", "", body)
    local = bool(re.search(r"\bin\s*$", body))
    if local:
        body = re.sub(r"\bin\s*$", "", body).strip()
    return local, _top_level_groups(body)


def _namespace(frames: list[_Frame]) -> str:
    names = [frame.name for frame in frames if frame.kind == "namespace" and frame.name]
    return ".".join(names)


def _apply_command(frames: list[_Frame], command: str) -> None:
    stripped = command.strip()
    if stripped.startswith("namespace "):
        frames.append(_Frame("namespace", stripped.split(None, 1)[1].strip()))
        return
    if re.match(r"^(?:noncomputable\s+)?section\b", stripped):
        parts = stripped.split(None, 1)
        name = None
        if parts[0] == "noncomputable":
            rest = parts[1].split(None, 1)
            name = rest[1].strip() if len(rest) > 1 else None
        elif len(parts) > 1:
            name = parts[1].strip() or None
        frames.append(_Frame("section", name))
        return
    if stripped == "mutual" or stripped.startswith("mutual "):
        frames.append(_Frame("mutual", None))
        return
    if stripped == "end" or stripped.startswith("end "):
        _pop_frame(
            frames, stripped[3:].strip() if stripped.startswith("end ") else None
        )
        return
    if stripped.startswith("omit"):
        frames[-1].omitted.update(_identifier_set(stripped[len("omit") :]))


def _pop_frame(frames: list[_Frame], name: str | None) -> None:
    if name is None:
        if len(frames) > 1:
            frames.pop()
        return
    target = name.split(".")[-1]
    for index in range(len(frames) - 1, 0, -1):
        frame = frames[index]
        frame_name = frame.name.split(".")[-1] if frame.name else None
        if frame.kind == "namespace" and frame_name == target:
            del frames[index:]
            return
        if frame.kind == "section" and frame.name == name:
            del frames[index:]
            return
    if len(frames) > 1:
        frames.pop()


def _consume_comment_block(lines: list[str], index: int) -> tuple[tuple[int, int], int]:
    text = "\n".join(lines[index:])
    end = _skip_block_comment(text, 0)
    consumed = text[:end].count("\n")
    return (index + 1, index + consumed + 1), index + consumed + 1


def _consume_attributes(lines: list[str], index: int) -> tuple[str, int]:
    chunks: list[str] = []
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped.startswith("@["):
            break
        _, next_index = _consume_balanced_lines(lines, index)
        chunks.append("\n".join(lines[index:next_index]))
        index = next_index
        while index < len(lines) and not lines[index].strip():
            index += 1
    return "\n".join(chunks), index


def _consume_balanced_lines(lines: list[str], index: int) -> tuple[str, int]:
    text = lines[index]
    depth = _bracket_depth(text)
    index += 1
    while depth > 0 and index < len(lines):
        text += "\n" + lines[index]
        depth += _bracket_depth(lines[index])
        index += 1
    return text, index


def _bracket_depth(text: str) -> int:
    depth = 0
    index = 0
    while index < len(text):
        if text.startswith("--", index):
            break
        if text.startswith("/-", index):
            index = _skip_block_comment(text, index)
            continue
        char = text[index]
        if char == '"':
            index = _skip_string(text, index)
            continue
        if char in _OPENERS:
            depth += 1
        elif char in _CLOSERS and depth:
            depth -= 1
        index += 1
    return depth


def _top_level_groups(text: str) -> list[str]:
    groups: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char in _OPENERS:
            end = _find_close(text, index)
            groups.append(text[index : end + 1])
            index = end + 1
            continue
        index += 1
    return groups


def _bound_names(rest: str) -> set[str]:
    names: set[str] = set()
    for group in _signature_groups(rest):
        names.update(_binder_names(group))
    return names


def _signature_groups(text: str) -> list[str]:
    groups: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        if text.startswith("--", index):
            newline = text.find("\n", index)
            index = length if newline < 0 else newline
            continue
        if text.startswith("/-", index):
            index = _skip_block_comment(text, index)
            continue
        char = text[index]
        if char == '"':
            index = _skip_string(text, index)
            continue
        if char in _OPENERS:
            end = _find_close(text, index)
            groups.append(text[index : end + 1])
            index = end + 1
            continue
        if char == ":" and not text.startswith(":=", index):
            break
        index += 1
    return groups


def _binder_names(group: str) -> list[str]:
    if not group or group[0] not in _OPENERS:
        return []
    inner = group[1:-1].strip()
    depth = 0
    head = inner
    index = 0
    while index < len(inner):
        char = inner[index]
        if char in _OPENERS:
            depth += 1
        elif char in _CLOSERS and depth:
            depth -= 1
        elif char == ":" and depth == 0 and not inner.startswith(":=", index):
            head = inner[:index]
            break
        index += 1
    if group.startswith("[") and ":" not in head:
        return []
    return _identifiers(head)


def _identifier_set(text: str) -> set[str]:
    return set(_identifiers(text))


def _identifiers(text: str) -> list[str]:
    found: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        if text.startswith("--", index):
            newline = text.find("\n", index)
            index = length if newline < 0 else newline
            continue
        if text.startswith("/-", index):
            index = _skip_block_comment(text, index)
            continue
        if text[index] == '"':
            index = _skip_string(text, index)
            continue
        if text.startswith("«", index):
            end = text.find("»", index + 1)
            if end < 0:
                break
            found.append(text[index : end + 1])
            index = end + 1
            continue
        if _is_ident_start(text[index]):
            end = index + 1
            while end < length and _is_ident_continue(text[end]):
                end += 1
            found.append(text[index:end])
            index = end
            continue
        index += 1
    return found


def _is_ident_start(char: str) -> bool:
    if char == "_":
        return True
    return unicodedata.category(char).startswith("L")


def _is_ident_continue(char: str) -> bool:
    if char in {"_", "'", "!", "?"}:
        return True
    category = unicodedata.category(char)
    return category.startswith(("L", "N", "M"))


def _find_close(text: str, open_index: int) -> int:
    closer = _OPENERS[text[open_index]]
    depth = 0
    index = open_index
    while index < len(text):
        if text.startswith("--", index):
            newline = text.find("\n", index)
            index = len(text) if newline < 0 else newline
            continue
        if text.startswith("/-", index):
            index = _skip_block_comment(text, index)
            continue
        char = text[index]
        if char == '"':
            index = _skip_string(text, index)
            continue
        if char in _OPENERS:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return index
        elif char in _CLOSERS and depth:
            depth -= 1
        index += 1
    return len(text) - 1


def _skip_block_comment(text: str, index: int) -> int:
    depth = 0
    cursor = index
    while cursor < len(text):
        if text.startswith("/-", cursor):
            depth += 1
            cursor += 2
            continue
        if text.startswith("-/", cursor):
            depth -= 1
            cursor += 2
            if depth == 0:
                return cursor
            continue
        cursor += 1
    return len(text)


def _skip_string(text: str, index: int) -> int:
    cursor = index + 1
    while cursor < len(text):
        if text[cursor] == "\\":
            cursor += 2
            continue
        if text[cursor] == '"':
            return cursor + 1
        cursor += 1
    return len(text)


def _strip_comments(text: str) -> str:
    chunks: list[str] = []
    index = 0
    while index < len(text):
        if text.startswith("--", index):
            newline = text.find("\n", index)
            if newline < 0:
                break
            index = newline
            continue
        if text.startswith("/-", index):
            index = _skip_block_comment(text, index)
            chunks.append(" ")
            continue
        chunks.append(text[index])
        index += 1
    return "".join(chunks)
