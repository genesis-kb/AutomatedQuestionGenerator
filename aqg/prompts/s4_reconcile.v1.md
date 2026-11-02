Verify the uncertain facts of this coding spec against the numbered reference passages.

Spec (JSON):
$spec

Uncertain facts:
$facts

Reference passages:
$passages

For EACH uncertain fact output one entry in `facts`:
- status "confirmed" if a passage states it, "corrected" if a passage contradicts it (put the right value in `corrected_value`), else "unverified".
- `reference_index`: the passage number you relied on, or -1.

Also extract up to 8 `test_vectors` that appear LITERALLY in the passages and fit the function signature exactly:
  signature: $signature
  `args`: one string per parameter, in order, formatted for its stdin line (integers in decimal, lists space-separated, booleans true/false, hex lowercase).
  `expected`: the printed return value in the same format.
Only include vectors whose input AND output are both given in a passage. Never compute or invent vectors. An empty list is fine.
