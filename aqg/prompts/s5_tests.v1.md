Design TEST INPUTS for this exercise. Do not compute expected outputs - a reference solution will be executed to produce them.

Spec (JSON):
$spec

Function (Python view): $signature
Parameters in stdin order:
$params

Each test input is a JSON object mapping EVERY parameter name ($param_names) to its argument string, e.g. $example
Formatting of each argument string:
- int / u64: decimal digits (u64: 0..18446744073709551615, int: |x| <= 9007199254740991)
- bool: true or false
- str: the raw string on one line (e.g. lowercase hex)
- list[...]: elements separated by single spaces; "" for an empty list

Return:
- `samples`: 1-3 small, illustrative inputs shown to users (objects as described above).
- `edge_cases`: 8-15 inputs covering boundaries (zero, maxima, thresholds, empty values, odd lengths, etc.). All inputs must be VALID per the spec.
- `generator_code`: Python code defining `def generate(rng):` that returns ONE random valid argument list as Python values (int, bool, str, or list), one per parameter, using only the `rng` (a random.Random) argument and the standard library. Keep values moderate in size.
$previous_attempt
