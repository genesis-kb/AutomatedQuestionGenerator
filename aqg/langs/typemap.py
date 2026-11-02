"""Closed type set -> stdin/stdout format and per-language code generation.

I/O convention (identical for every language, so expected outputs are shared):
  * stdin line i holds parameter i.
  * int / u64: decimal.  bool: `true` / `false`.  str: the whole line (no newline).
  * list[T]: elements separated by single spaces; an empty line is an empty list.
  * The return value is printed on one line with the same formatting.

Wrappers are generated here, deterministically, never by the LLM. They are appended
to the user's code by hello-bitcoin (user + separator + wrapper), so every wrapper
identifier is prefixed with `__aqg` and the Rust wrapper avoids `use` lines (a
duplicate `use` in user code would be a compile error).
"""

from __future__ import annotations

import json
from typing import Any

from aqg.schemas.spec import CodingSpec

INT_MAX = 2**53 - 1  # exact in JavaScript Number
U64_MAX = 2**64 - 1

_SCALARS = ("int", "u64", "bool", "str")


class ValueError_(ValueError):
    pass


def elem_type(t: str) -> str | None:
    return t[5:-1] if t.startswith("list[") else None


def to_camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(p[:1].upper() + p[1:] for p in rest if p)


# ── host-side parsing / formatting (used to validate tests) ───────────────────


def _parse_scalar(t: str, token: str) -> Any:
    if t in ("int", "u64"):
        token = token.strip()
        if not token or not (token.lstrip("-").isdigit()):
            raise ValueError_(f"{token!r} is not a decimal integer")
        value = int(token)
        if t == "int" and abs(value) > INT_MAX:
            raise ValueError_(f"int {value} outside +/-(2^53-1)")
        if t == "u64" and not 0 <= value <= U64_MAX:
            raise ValueError_(f"u64 {value} outside 0..2^64-1")
        return value
    if t == "bool":
        token = token.strip()
        if token not in ("true", "false"):
            raise ValueError_(f"bool must be 'true' or 'false', got {token!r}")
        return token == "true"
    if t == "str":
        if "\n" in token or "\r" in token:
            raise ValueError_("str must not contain newlines")
        return token
    raise ValueError_(f"unknown type {t}")


def parse_line(t: str, line: str) -> Any:
    inner = elem_type(t)
    if inner is None:
        return _parse_scalar(t, line)
    return [_parse_scalar(inner, tok) for tok in line.split()]


def format_value(t: str, value: Any) -> str:
    inner = elem_type(t)
    if inner is not None:
        if not isinstance(value, (list, tuple)):
            raise ValueError_(f"expected a list for {t}, got {type(value).__name__}")
        parts = [format_value(inner, v) for v in value]
        if inner == "str" and any((not p) or any(c.isspace() for c in p) for p in parts):
            raise ValueError_("list[str] elements must be non-empty and contain no whitespace")
        return " ".join(parts)
    if t == "bool":
        if not isinstance(value, bool):
            raise ValueError_(f"expected bool, got {value!r}")
        return "true" if value else "false"
    if t in ("int", "u64"):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError_(f"expected integer for {t}, got {value!r}")
        _parse_scalar(t, str(value))  # range check
        return str(value)
    if t == "str":
        if not isinstance(value, str):
            raise ValueError_(f"expected str, got {value!r}")
        _parse_scalar("str", value)
        return value
    raise ValueError_(f"unknown type {t}")


def normalize_args(spec: CodingSpec, args: list[str] | dict[str, str]) -> str:
    """Validate raw per-parameter lines (positional list or {param_name: value}) -> canonical stdin text."""
    if isinstance(args, dict):
        names = [p.name for p in spec.params]
        missing = [n for n in names if n not in args]
        extra = [k for k in args if k not in names]
        if missing or extra:
            raise ValueError_(f"arguments must have exactly the keys {names}; missing {missing}, unexpected {extra}")
        args = [args[n] for n in names]
    if len(args) != len(spec.params):
        raise ValueError_(f"expected {len(spec.params)} argument line(s), got {len(args)}")
    lines = []
    for param, raw in zip(spec.params, args):
        if not isinstance(raw, str):
            raw = json.dumps(raw) if not isinstance(raw, (int, float)) else str(raw)
        try:
            value = parse_line(param.type, raw)
        except ValueError_ as exc:
            raise ValueError_(f"param {param.name}: {exc}") from exc
        lines.append(format_value(param.type, value))
    return "\n".join(lines) + "\n"


def values_to_stdin(spec: CodingSpec, values: list[Any]) -> str:
    """Python values (e.g. from the random generator) -> canonical stdin text."""
    if not isinstance(values, list) or len(values) != len(spec.params):
        raise ValueError_(f"generator must return a list of {len(spec.params)} values, got {values!r}")
    lines = []
    for param, value in zip(spec.params, values):
        try:
            lines.append(format_value(param.type, value))
        except ValueError_ as exc:
            raise ValueError_(f"param {param.name}: {exc}") from exc
    return "\n".join(lines) + "\n"


def canonical_output(text: str) -> str:
    """What the default checker compares: stripped lines; we store them joined by \\n."""
    return "\n".join(line.strip() for line in text.strip().splitlines())


# ── per-language type names ───────────────────────────────────────────────────

PY_TYPES = {"int": "int", "u64": "int", "bool": "bool", "str": "str"}
JS_TYPES = {"int": "number", "u64": "bigint", "bool": "boolean", "str": "string"}
RS_TYPES = {"int": "i64", "u64": "u64", "bool": "bool", "str": "String"}
RS_DEFAULTS = {"int": "0", "u64": "0", "bool": "false", "str": "String::new()"}


def py_type(t: str) -> str:
    inner = elem_type(t)
    return f"list[{PY_TYPES[inner]}]" if inner else PY_TYPES[t]


def js_type(t: str) -> str:
    inner = elem_type(t)
    return f"{JS_TYPES[inner]}[]" if inner else JS_TYPES[t]


def rs_type(t: str) -> str:
    inner = elem_type(t)
    return f"Vec<{RS_TYPES[inner]}>" if inner else RS_TYPES[t]


def rs_default(t: str) -> str:
    return "Vec::new()" if elem_type(t) else RS_DEFAULTS[t]


def function_name(spec: CodingSpec, language: str) -> str:
    return to_camel(spec.function_name) if language == "javascript" else spec.function_name


def param_names(spec: CodingSpec, language: str) -> list[str]:
    return [to_camel(p.name) if language == "javascript" else p.name for p in spec.params]


def signature(spec: CodingSpec, language: str) -> str:
    fn = function_name(spec, language)
    names = param_names(spec, language)
    if language == "python3":
        params = ", ".join(f"{n}: {py_type(p.type)}" for n, p in zip(names, spec.params))
        return f"def {fn}({params}) -> {py_type(spec.return_type)}"
    if language == "javascript":
        return f"function {fn}({', '.join(names)})"
    params = ", ".join(f"{n}: {rs_type(p.type)}" for n, p in zip(names, spec.params))
    return f"fn {fn}({params}) -> {rs_type(spec.return_type)}"


# ── starters ──────────────────────────────────────────────────────────────────


def starter(spec: CodingSpec, language: str, helpers_code: str = "") -> str:
    prefix = f"{helpers_code.rstrip()}\n\n\n" if helpers_code.strip() else ""
    if language == "python3":
        return f"{prefix}{signature(spec, language)}:\n    # YOUR CODE HERE\n    pass\n"
    if language == "javascript":
        doc = ["/**"]
        for name, p in zip(param_names(spec, language), spec.params):
            doc.append(f" * @param {{{js_type(p.type)}}} {name}")
        doc.append(f" * @returns {{{js_type(spec.return_type)}}}")
        doc.append(" */")
        return f"{prefix}" + "\n".join(doc) + f"\n{signature(spec, language)} {{\n    // YOUR CODE HERE\n}}\n"
    return (f"{prefix}#[allow(unused_variables)]\n{signature(spec, language)} {{\n"
            f"    // YOUR CODE HERE\n    {rs_default(spec.return_type)}\n}}\n")


# ── wrappers ──────────────────────────────────────────────────────────────────


def _py_parse(t: str, line: str) -> str:
    inner = elem_type(t)
    if inner is None:
        return {"int": f"int({line}.strip())", "u64": f"int({line}.strip())",
                "bool": f"({line}.strip() == 'true')", "str": line}[t]
    conv = {"int": "int(__x)", "u64": "int(__x)", "bool": "(__x == 'true')", "str": "__x"}[inner]
    return f"[{conv} for __x in {line}.split()]"


def _py_format(t: str, value: str) -> str:
    inner = elem_type(t)
    scalar = {"int": "str(int({v}))", "u64": "str(int({v}))",
              "bool": "('true' if {v} else 'false')", "str": "str({v})"}
    if inner is None:
        return scalar[t].format(v=value)
    return f"' '.join({scalar[inner].format(v='__x')} for __x in {value})"


def _js_parse(t: str, line: str) -> str:
    inner = elem_type(t)
    if inner is None:
        return {"int": f"Number({line}.trim())", "u64": f"BigInt({line}.trim())",
                "bool": f"({line}.trim() === 'true')", "str": line}[t]
    conv = {"int": "Number", "u64": "BigInt", "bool": "((__x) => __x === 'true')", "str": "String"}[inner]
    return f"({line}.trim() === '' ? [] : {line}.trim().split(/\\s+/).map({conv}))"


def _js_format(t: str, value: str) -> str:
    inner = elem_type(t)
    scalar = {"int": "String({v})", "u64": "String({v})", "bool": "({v} ? 'true' : 'false')", "str": "String({v})"}
    if inner is None:
        return scalar[t].format(v=value)
    return f"Array.from({value}).map((__x) => {scalar[inner].format(v='__x')}).join(' ')"


def _rs_parse(t: str, line: str) -> str:
    inner = elem_type(t)
    if inner is None:
        return {"int": f"{line}.trim().parse::<i64>().expect(\"invalid int\")",
                "u64": f"{line}.trim().parse::<u64>().expect(\"invalid u64\")",
                "bool": f"({line}.trim() == \"true\")",
                "str": f"{line}.to_string()"}[t]
    conv = {"int": "__x.parse::<i64>().expect(\"invalid int\")", "u64": "__x.parse::<u64>().expect(\"invalid u64\")",
            "bool": "__x == \"true\"", "str": "__x.to_string()"}[inner]
    return f"{line}.split_whitespace().map(|__x| {conv}).collect::<Vec<{RS_TYPES[inner]}>>()"


def _rs_format(t: str, value: str) -> str:
    inner = elem_type(t)
    if inner is None:
        if t == "bool":
            return f"(if {value} {{ \"true\" }} else {{ \"false\" }}).to_string()"
        return f"format!(\"{{}}\", {value})"
    if inner == "bool":
        conv = "(if *__x { \"true\" } else { \"false\" }).to_string()"
    else:
        conv = "__x.to_string()"
    return f"{value}.iter().map(|__x| {conv}).collect::<Vec<String>>().join(\" \")"


def wrapper(spec: CodingSpec, language: str) -> str:
    fn = function_name(spec, language)
    n = len(spec.params)
    if language == "python3":
        lines = [
            "import sys as __aqg_sys",
            "__aqg_lines = __aqg_sys.stdin.read().split('\\n')",
            "",
            "def __aqg_line(i):",
            "    return __aqg_lines[i].rstrip('\\r') if i < len(__aqg_lines) else ''",
            "",
            "__aqg_args = [",
            *[f"    {_py_parse(p.type, f'__aqg_line({i})')}," for i, p in enumerate(spec.params)],
            "]",
            f"__aqg_result = {fn}(*__aqg_args)",
            f"print({_py_format(spec.return_type, '__aqg_result')})",
        ]
        return "\n".join(lines) + "\n"
    if language == "javascript":
        lines = [
            "const __aqgLines = require('fs').readFileSync(0, 'utf-8').split('\\n');",
            "function __aqgLine(i) { return i < __aqgLines.length ? __aqgLines[i].replace(/\\r$/, '') : ''; }",
            f"const __aqgResult = {fn}(",
            *[f"    {_js_parse(p.type, f'__aqgLine({i})')}," for i, p in enumerate(spec.params)],
            ");",
            f"console.log({_js_format(spec.return_type, '__aqgResult')});",
        ]
        return "\n".join(lines) + "\n"
    args = ", ".join(f"__aqg_a{i}" for i in range(n))
    lines = [
        "fn __aqg_line<'a>(lines: &[&'a str], i: usize) -> &'a str {",
        "    if i < lines.len() { lines[i] } else { \"\" }",
        "}",
        "",
        "fn main() {",
        "    let mut __aqg_input = String::new();",
        "    std::io::Read::read_to_string(&mut std::io::stdin(), &mut __aqg_input).expect(\"stdin\");",
        "    let __aqg_lines: Vec<&str> = __aqg_input.split('\\n').map(|l| l.trim_end_matches('\\r')).collect();",
        *[f"    let __aqg_a{i} = {_rs_parse(p.type, f'__aqg_line(&__aqg_lines, {i})')};"
          for i, p in enumerate(spec.params)],
        f"    let __aqg_result = {fn}({args});",
        f"    println!(\"{{}}\", {_rs_format(spec.return_type, '__aqg_result')});",
        "}",
    ]
    return "\n".join(lines) + "\n"


def io_format_markdown(spec: CodingSpec) -> tuple[str, str]:
    """(input section, output section) for the problem statement."""
    def describe(t: str) -> str:
        inner = elem_type(t)
        base = {"int": "an integer", "u64": "an unsigned 64-bit integer", "bool": "`true` or `false`",
                "str": "a string"}
        if inner:
            return f"space-separated values, each {base[inner]} (an empty line means an empty list)"
        return base[t]

    inp = ["The judge reads stdin for you and calls your function. Line by line:", ""]
    for i, p in enumerate(spec.params, 1):
        extra = f" ({p.constraints})" if p.constraints else ""
        inp.append(f"{i}. `{p.name}`: {describe(p.type)}: {p.description}{extra}")
    out = f"Return {describe(spec.return_type)}: {spec.return_description}. The judge prints it on one line."
    return "\n".join(inp), out
