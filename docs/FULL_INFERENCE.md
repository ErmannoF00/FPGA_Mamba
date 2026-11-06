# Complete fixed-point inference implementation

The classifier consumes a full 8 × 8 signed Q6.10 image and produces ten
signed Q6.10 logits and a four-bit class index in SystemVerilog. The host does
not supply any hidden activations, selective-scan coefficients, or projected
tokens. All weights are converted to ROMs from the trained bilinear checkpoint.

## Dataflow

1. Image input handshake (1,024-bit frame, 64 signed Q6.10 pixels).
2. Input projection and gating projection, 8 × 8 each.
3. Causal three-tap depthwise convolution and ReLU.
4. Learned B/C/delta projections and four-level delta selection.
5. ROM-based bilinear coefficients and per-state recurrent arithmetic, via
   `precision_scan.sv`, with the existing 8 × 4 precision map.
6. Output gating, output projection and residual addition.
7. Flatten eight 8-element token outputs and evaluate the learned 64 × 10 head.
8. Signed-logit argmax with deterministic low-index tie breaking.

The current `mamba_full_top.sv` processes one token in two clock cycles. It
prioritizes numerical correctness over physical PPA; the front-end and head
are combinational, so FPGA frequency and resource costs remain unmeasured.
Calling this timing-closed or energy-efficient hardware would be incorrect.

## Numerical semantics

Weights and activations are signed Q6.10. All products are rounded to nearest
even at the operator boundaries used by `model.full_integer`; intermediate
outputs are clamped to signed 16-bit values. Every recurrent state has six
signed integer bits plus its calibrated number of fractional bits. Bilinear
coefficient ROMs are generated from the learned diagonal poles and quantized
time-step levels; no runtime division or exponential is required.

Full fixed-point inference is **not bitwise identical to the original
floating-weight QAT model**. The complete integer model maintains 346/360
correct predictions versus 346/360 in the original model, with 355/360
predictions agreeing, on the held-out sklearn digits set. These are model
comparisons; RTL equivalence requires running both simulators.

## Verification

```bash
make -f Makefile.full full-reference
make -f Makefile.full full-lint
make -f Makefile.full full-iverilog
make -f Makefile.full full-verilator
make -f Makefile.full full-synth
```

Only `full-iverilog` and `full-verilator` validate image-to-logit hardware.
Both compare all ten logits exactly for 360 held-out images against a
separately evaluated integer model. `full-synth` is technology-independent
structural synthesis, not FPGA placement, timing closure or board execution.

## Boundaries of the claim

The architecture is **Mamba-derived**, not an official Mamba-1 or Mamba-2
kernel reproduction. Bilinear discretization, QAT, and offline precision
allocation are not independently novel results. Future research concerns
runtime precision adaptation with conservative error accounting and actual
hardware-exploitable narrow arithmetic; those capabilities are not present
in this current full-inference reference design.
