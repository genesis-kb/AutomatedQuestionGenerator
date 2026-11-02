"""Static checks on LLM-written solution code before it reaches the sandbox.

Enforces the plan's feasibility rules: standard library only, no I/O (the wrapper
owns stdin/stdout), no `main` (the Rust wrapper defines it).
"""

from __future__ import annotations

import ast
import re
import sys

NODE_BUILTINS = {"crypto", "buffer", "util", "assert", "string_decoder"}
RUST_ALLOWED_ROOTS = {"std", "core", "alloc", "self", "super", "crate"}
PY_BANNED_MODULES = {"subprocess", "socket", "os", "shutil", "ctypes", "multiprocessing", "threading", "asyncio"}


def strip_code_fences(code: str) -> str:
    """LLMs often wrap code in ``` fences even when asked not to."""
    text = code.strip()
    match = re.match(r"^```[a-zA-Z0-9_+-]*\s*\n(.*?)\n?```\s*$", text, re.S)
    return match.group(1) if match else text


def check_python(code: str, function_name: str) -> list[str]:
    problems: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"python syntax error: {exc}"]
    defined = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if function_name not in defined:
        problems.append(f"function `{function_name}` is not defined at top level")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for name in names:
                root = name.split(".")[0]
                if root not in sys.stdlib_module_names:
                    problems.append(f"non-stdlib import `{name}`")
                elif root in PY_BANNED_MODULES:
                    problems.append(f"forbidden module `{name}`")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"print", "input", "open", "exec", "eval"}:
            problems.append(f"call to `{node.func.id}()` is not allowed in a solution (the wrapper does I/O)")
        if isinstance(node, ast.Attribute) and node.attr in {"stdin", "stdout"}:
            problems.append("direct stdin/stdout access is not allowed in a solution")
    return problems


def check_javascript(code: str, function_name: str) -> list[str]:
    problems: list[str] = []
    if not re.search(rf"\bfunction\s+{re.escape(function_name)}\s*\(|\b(const|let|var)\s+{re.escape(function_name)}\s*=", code):
        problems.append(f"function `{function_name}` is not defined")
    for module in re.findall(r"require\(\s*['\"]([^'\"]+)['\"]\s*\)", code):
        if module.removeprefix("node:") not in NODE_BUILTINS:
            problems.append(f"require('{module}') is not allowed (only {sorted(NODE_BUILTINS)})")
    if re.search(r"^\s*import\s", code, re.M):
        problems.append("ES module `import` is not supported by the judge (CommonJS only)")
    for banned in ("console.log", "process.stdout", "process.stdin", "readFileSync", "process.exit"):
        if banned in code:
            problems.append(f"`{banned}` is not allowed in a solution (the wrapper does I/O)")
    if re.search(r"^\s*(const|let|var)\s+fs\b", code, re.M):
        problems.append("declaring `fs` would clash with common user code; not allowed")
    return problems


def check_rust(code: str, function_name: str) -> list[str]:
    problems: list[str] = []
    if not re.search(rf"\bfn\s+{re.escape(function_name)}\s*[<(]", code):
        problems.append(f"function `{function_name}` is not defined")
    if re.search(r"\bfn\s+main\s*\(", code):
        problems.append("`fn main` must not be defined (the hidden wrapper defines it)")
    if re.search(r"\bextern\s+crate\b", code):
        problems.append("`extern crate` is not available (the judge compiles with plain rustc)")
    for path in re.findall(r"^\s*use\s+([a-zA-Z_][a-zA-Z0-9_]*)", code, re.M):
        if path not in RUST_ALLOWED_ROOTS:
            problems.append(f"`use {path}::...` refers to an external crate; only std is available")
    for banned in ("println!", "print!", "eprintln!", "std::io::stdin", "stdin()"):
        if banned in code:
            problems.append(f"`{banned}` is not allowed in a solution (the wrapper does I/O)")
    return problems


def check_solution(language: str, code: str, function_name: str) -> list[str]:
    return {"python3": check_python, "javascript": check_javascript, "rust": check_rust}[language](code, function_name)
