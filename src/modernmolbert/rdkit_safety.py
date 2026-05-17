"""Small guards for RDKit parser entry points.

Some RDKit builds can abort the interpreter for malformed strings before Python
can catch an exception. These helpers only reject obviously non-SMILES text;
RDKit remains the source of truth for full chemical validation.
"""

_BARE_TWO_CHAR_ATOMS = {"Br", "Cl"}
_BARE_ONE_CHAR_ATOMS = set("BCNOPSFIbcnops")
_BRACKET_TWO_CHAR_ATOMS = {"Br", "Cl"}
_BRACKET_ONE_CHAR_ATOMS = set("HBCNOPSFIbcnops")
_STRUCTURAL_CHARS = set("()-.=#$:/\\+@0123456789")


def _has_safe_bracket_atom(body: str) -> bool:
    i = 0
    while i < len(body) and body[i].isdigit():
        i += 1

    if i >= len(body):
        return False

    if body[i : i + 2] in _BRACKET_TWO_CHAR_ATOMS:
        return True

    return body[i] in _BRACKET_ONE_CHAR_ATOMS


def looks_like_smiles(value: object) -> bool:
    """Return False for strings that are clearly outside basic SMILES syntax."""

    if value is None:
        return False

    text = str(value).strip()
    if not text:
        return False

    i = 0
    while i < len(text):
        ch = text[i]

        if ch == "[":
            end = text.find("]", i + 1)
            if end == -1 or end == i + 1:
                return False
            body = text[i + 1 : end]
            if "[" in body or not _has_safe_bracket_atom(body):
                return False
            i = end + 1
            continue

        if ch == "]":
            return False

        if text[i : i + 2] in _BARE_TWO_CHAR_ATOMS:
            i += 2
            continue

        if ch in _BARE_ONE_CHAR_ATOMS:
            i += 1
            continue

        if ch == "%":
            if i + 2 >= len(text) or not text[i + 1 : i + 3].isdigit():
                return False
            i += 3
            continue

        if ch in _STRUCTURAL_CHARS:
            i += 1
            continue

        return False

    return True
