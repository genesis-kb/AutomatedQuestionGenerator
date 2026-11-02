Turn this concept from a conference talk into a precise CODING SPEC for a judge exercise.

Concept: $concept
Why it is codeable: $why
Verbatim excerpt: "$excerpt"
Surrounding transcript (speech-to-text output; may contain recognition errors in numbers, hex and names):
"""
$context
"""

The exercise is ALWAYS "implement one pure function". Constraints on the spec:
- `function_name` and every `params[].name`: snake_case, descriptive, not a language keyword.
- Types must come from this closed set: int (|x| <= 2^53-1), u64 (0..2^64-1), bool, str (single line, e.g. lowercase hex), list[int], list[u64], list[str] (no spaces inside elements), list[bool].
- 1 to 6 params, but prefer 1-3: a focused function is far better than one that takes a whole transaction apart. Each param is given on its own stdin line by the judge; you only design the function.
- Keep it small enough for a 15-60 minute exercise. Prefer one crisp sub-step of the mechanism over a whole protocol.
- Must be solvable with standard libraries only. If hashing is needed, list it in `needs_helpers` (sha256, ripemd160, hex). No elliptic-curve or >64-bit integer arithmetic.
- `algorithm_outline`: numbered, exact steps (byte order, padding, prefixes, encodings).
- `constants`: every magic number / prefix / tag used, with source "transcript" (or "standard" if you are certain from the Bitcoin spec).
- `uncertain_facts`: EVERY fact that a speech-to-text error or a speaker slip could have corrupted (constants, byte orders, thresholds, BIP numbers). These will be verified against BIPs.
- `search_queries`: 2-4 short queries to verify those facts in BIPs / Bitcoin docs (e.g. "BIP 141 witness serialization marker flag").
- `difficulty`: easy | medium | hard.
