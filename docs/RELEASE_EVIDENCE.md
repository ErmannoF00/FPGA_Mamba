# Release evidence and scope — November 2026

## Observed results (from the user's WSL logs)

| Test | Observation | Limits |
|---|---|---|
| Python unit tests | 18 tests passed in this review environment; user's 11 classifier/research tests passed | Not RTL execution |
| Trained bilinear model | 96.11% / 360 held-out images | PyTorch accuracy |
| ZOH control | 95.56% / 360 held-out images | Different trained checkpoint |
| Scan simulation | 712 cycle comparisons PASS in Icarus and Verilator | Front end/head still in PyTorch |
| Icarus trace capture | 400 consumed 128-bit scan outputs | 50 held-out images × 8 tokens |
| Icarus RTL-fed classifier | 49/50 correct; 0 max logit difference vs integer reference | Hybrid, not on-board FPGA |
| Yosys scan-only synthesis | 159,012 generic cells; 0 structural problems; 641 FF primitives | No target mapping, no timing, no power |
| Runtime adaptive precision | Not implemented | State widths selected at elaboration |
| FPGA device execution | Not performed | No board or implementation reports |

## What is distinctive

A trainable, compact Mamba-derived classifier is coupled to an RTL scan with **individually sized state registers** using a calibration-derived precision map, comparing a bilinear discretization against a ZOH baseline. The current bit allocation is decided **offline**, not dynamically per token. The scan operates on coefficients exported by the software model, and the final classifier is executed on the host. The originality of offline per-pole width allocation has **not** been established against current literature. Avoid calling it a novel algorithm without systematic baselines and a literature review.

## Mandatory release gates

1. `python3 -m unittest discover -s tests -v` — software contract.
2. `make -f Makefile.research release-check` — lint, independent simulator traces, classifier injection, numerical checks and generic synthesis.
3. `make reference lint directed verilator iverilog` — retained scalar-scan regression. The subsequent WSL rerun passed three seeds in both simulators (7,200 cycle comparisons per simulator, four lanes).
4. `git diff --check`, `git status --short`, inspect every deletion (notably old MNIST modules); verify `.gitignore` excludes generated artifacts.
5. Commit only intended source, documentation and small redistributable weights; do not commit `build/`, core dumps, datasets without license, or private files.

## What is required before claiming full FPGA inference

- Quantized matrix-multiply hardware for input/gate/B/C/Δ/output projections and the classifier head.
- Causal convolution, activation and gating hardware; coefficient lookup for bilinear poles.
- Weight/activation storage interfaces and a controller scheduling complete sequences.
- Bit-exact full-model RTL-to-software comparison, followed by FPGA-specific synthesis, place-and-route, timing closure and device execution.
- Measurements against fixed-Q6.10 and equal-bit-budget alternatives on the **same device** before speed/energy/resource reduction claims.

## Research direction after this release

A runtime error-budgeted precision controller is an attractive follow-up, but a compile-time width map does not implement it. A meaningful experiment requires: a formally specified quantization-error invariant, synthesizable mode selection, **physically narrower arithmetic**, and measured comparisons against static allocation and simpler heuristics. The scope of the current repository should not be expanded in its README before those experiments exist.
