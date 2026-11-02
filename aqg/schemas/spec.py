"""S3 contract: a language-neutral coding spec.

Problems are always "implement one pure function". Parameter and return types come
from a closed set so wrappers/starters can be generated deterministically for every
judge language (see aqg/langs/typemap.py).
"""

from __future__ import annotations

import keyword
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

ValueType = Literal["int", "u64", "bool", "str", "list[int]", "list[u64]", "list[str]", "list[bool]"]
Helper = Literal["sha256", "ripemd160", "hex"]

_IDENT = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
_RESERVED = set(keyword.kwlist) | {
    "main", "input", "print", "fn", "let", "mut", "impl", "struct", "use", "mod", "match", "loop", "type",
    "function", "var", "const", "new", "delete", "this", "class", "self", "super", "crate", "ref", "move",
    "static", "enum", "trait", "where", "async", "await", "dyn", "unsafe", "extern", "pub", "string",
}


class Param(BaseModel):
    name: str = Field(description="snake_case identifier")
    type: ValueType
    description: str
    constraints: str = Field(default="", description="Value ranges / format, e.g. '0 <= n < 2^64', 'lowercase hex'")

    @field_validator("name")
    @classmethod
    def _ident(cls, v: str) -> str:
        v = v.strip()
        if not _IDENT.match(v) or v in _RESERVED:
            raise ValueError(f"parameter name {v!r} must be snake_case and not a reserved word")
        return v


class Constant(BaseModel):
    name: str
    value: str
    source: str = "transcript"


class CodingSpec(BaseModel):
    title_hint: str
    problem_kind: Literal["encode", "decode", "parse", "validate", "compute", "hash", "serialize"]
    function_name: str = Field(description="snake_case function name")
    params: list[Param] = Field(min_length=1, max_length=6)
    return_type: ValueType
    return_description: str
    algorithm_outline: list[str] = Field(min_length=1)
    constants: list[Constant] = Field(default_factory=list)
    edge_cases: list[str] = Field(default_factory=list)
    uncertain_facts: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list, max_length=4)
    needs_helpers: list[Helper] = Field(default_factory=list)
    difficulty: Literal["easy", "medium", "hard"] = "medium"

    @field_validator("function_name")
    @classmethod
    def _fn(cls, v: str) -> str:
        v = v.strip()
        if not _IDENT.match(v) or v in _RESERVED:
            raise ValueError(f"function_name {v!r} must be snake_case and not a reserved word")
        return v

    @model_validator(mode="after")
    def _unique_params(self) -> "CodingSpec":
        names = [p.name for p in self.params]
        if len(names) != len(set(names)):
            raise ValueError("parameter names must be unique")
        if self.function_name in names:
            raise ValueError("function_name must differ from parameter names")
        return self
