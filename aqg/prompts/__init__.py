"""Versioned prompt templates (`<name>.v<N>.md`, `$var` placeholders).

The sha256 of every rendered template is recorded in the run manifest so each
output can be traced back to the exact prompt text that produced it.
"""

from __future__ import annotations

import hashlib
from importlib import resources
from string import Template

VERSIONS = {
    "system": "v1",
    "s2_triage": "v1",
    "s2_extract": "v1",
    "s3_spec": "v1",
    "s4_reconcile": "v1",
    "s5_statement": "v1",
    "s5_reference": "v1",
    "s5_tests": "v1",
}


def template_text(name: str) -> str:
    return resources.files("aqg.prompts").joinpath(f"{name}.{VERSIONS[name]}.md").read_text(encoding="utf-8")


def template_hash(name: str) -> str:
    return "sha256:" + hashlib.sha256(template_text(name).encode()).hexdigest()[:16]


def render(name: str, **values: object) -> str:
    return Template(template_text(name)).safe_substitute({k: str(v) for k, v in values.items()})


def system_prompt() -> str:
    return template_text("system")


def all_hashes() -> dict[str, str]:
    return {f"{name}.{version}": template_hash(name) for name, version in VERSIONS.items()}
