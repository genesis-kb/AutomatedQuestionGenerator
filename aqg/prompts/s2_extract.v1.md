From the transcript CHUNK below, extract mechanisms that can become an auto-graded coding exercise of this shape:

  "Implement one pure function f(inputs) -> output" graded by exact output match.
  Inputs/outputs are small values: integers, booleans, short strings (e.g. lowercase hex), or space-separated lists of those.
  It must be solvable in Python, JavaScript and Rust using ONLY standard libraries
  (SHA-256, RIPEMD-160 and hex helpers are provided; there is NO big-integer / secp256k1 elliptic-curve support).

Rules:
- Only deterministic mechanisms (same input -> same output). No opinions, history, economics, UI, or "explain why" topics.
- `excerpt` MUST be copied verbatim from the chunk (max 300 characters) - the sentence(s) where the mechanism is mentioned.
- `codeability`: 0-5, how precisely the mechanism can be specified as such a function.
- `requires_unsupported`: true ONLY if it needs real elliptic-curve math, integers beyond 64 bits, network access, or external libraries. Serialization, parsing, hashing, encoding and fee/weight arithmetic are all supported (false).
- `concept`: name the mechanism itself (e.g. "Serialize a transaction input"), not a slide heading like "Step 4".
- Returning an empty list is correct when nothing qualifies.

Talk title: $title
CHUNK:
"""
$chunk
"""
