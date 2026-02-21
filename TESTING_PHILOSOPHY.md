# Testing Philosophy

## Core Rule
All statistical detector tests must use **controlled synthetic arrays** where
the property being tested is guaranteed by construction — not by running an
embedder on a real image.

Real images have too many uncontrolled variables to serve as reliable unit
test inputs. A test that passes on a real image today may fail tomorrow if
the image is replaced, and a failure gives no clear signal about which
property broke.

## What This Means in Practice
- Chi-square tests: construct arrays with known pair distributions
- Entropy tests: construct bit arrays with known entropy values
- RS tests: construct smooth gradient arrays with known smoothness properties
- Histogram tests: construct arrays with known even/odd pair imbalances

## Real Images in Tests
Real images (including the Mandrill) may be used in:
- Integration tests (full pipeline smoke tests)
- Benchmark evaluation (Phase 6)
- Visual verification (not assertions)

Real images must NEVER be the sole basis for asserting correctness of a
statistical property.

## Chi-Square Probability Direction
probability = p_value (NOT 1 - p_value)
A HIGH p-value means the test failed to reject pair equality — which IS the
stego signal. This must never be inverted.
