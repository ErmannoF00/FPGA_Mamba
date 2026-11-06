# State precision experiment

## Research question

For a fixed signed integer range, does allocating recurrent state fractional bits by pole persistence and output sensitivity reduce distortion relative to uniform allocation under an equal storage budget?

This is not a claim that mixed-precision SSM computation, bilinear discretization, or quantization-aware training is itself original. The contribution to evaluate is the specific state-allocation methodology, compile-time state-register specialization, and its trade-off against network-level output error after synthesis.

## Current model

A trained 8-channel, 4-state-per-channel classifier uses a four-bin time-step codebook and the bilinear transform. The original model operates with 16-bit Q6.10 internal fake quantization. Only recurrent state storage precision is varied in the initial experiments; all other arithmetic remains Q6.10.

The pole-conditioned allocator uses one-pole-at-a-time logit perturbations measured exclusively on the validation set, multiplied by a weak theoretical lifetime weight `1/(1-rho)`. A dynamic program minimizes the sum of these proxy costs at an exact total fractional-bit budget of 64. This is a heuristic that must be compared against stronger allocators and joint QAT.

## Reproducibility

- Deterministic seed 19; stratified train/validation/test split.
- Compare bilinear and ZOH models with independent training under identical optimizer settings.
- Evaluate precision maps on the same held-out 360 samples, and report both accuracy and numerical distortion.
- Compare uniform 2 fractional bits per state to an equal-total-budget mixed map.
- Reproduce integer recurrent scan outputs from model-derived coefficients before exercising RTL.
- Run both Icarus Verilog and Verilator before accepting hardware equivalence.

## Preliminary findings

The included digit classifier obtains 96.11% test accuracy. Precision calibration reduces test logit MSE from 0.3893 to 0.3110 under an equal state-storage budget (a 20.1% reduction). That is not an improvement in test classification accuracy and does not yet establish resource or energy savings. A 50-example replay corpus exercises 400 recurrent scan tokens, including stalls and sequence boundaries. Its RTL simulation still requires confirmation on a machine with HDL tools.

## Next architectural implementation

1. Make the coefficient generation path synthesizable: time-step bin, precomputed Tustin `(alpha,gain)` entries, B/C projections and excitation.
2. Introduce a tiled BRAM-based weight store and a single configurable projection MAC engine; measure bottlenecks rather than fully unrolling every matrix.
3. Integrate causal convolution, gate, output projection and 10-class head into a clocked top module.
4. Export the exact trained parameter image and compare full-network logits and predictions against integer and PyTorch references.
5. Route both fixed-precision and mixed-precision variants to the *same* FPGA target; report FF/LUT/DSP/BRAM usage, critical path, achieved frequency, latency and measured dynamic power where available.
6. Extend to longer-sequence data and test whether the precision benefit persists as `rho` approaches one.

## Novelty and falsification

If mixed precision has greater hardware cost, changes predictions unacceptably, fails on longer sequences, or duplicates prior methods without a technical distinction, it is not a publishable improvement. Include these negative results rather than excluding them. Comparisons against uniform INT8/INT16 and existing quantization-focused Mamba FPGA accelerators are mandatory.
