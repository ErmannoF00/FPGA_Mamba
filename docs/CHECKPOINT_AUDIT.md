# Checkpoint-derived integer inference

- Source checkpoint: `weights/digits_bilinear.pt`.
- SHA-256: `e73c5d82e836415fa9f15adafa61f05ada6f880d362812513ec29e34337a2bfa`.
- Test set: 360 held-out images from sklearn digits, fixed stratified split.
- Floating-weight PyTorch predictions correct: 346/360.
- Quantized-weight, all-integer reference predictions correct: 346/360.
- Prediction agreement: 355/360.
- Ten signed Q6.10 logits are compared, not only argmax labels.

The arithmetic oracle is independent of PyTorch's `forward()` and implements
nearest-even rounding and saturating 16-bit arithmetic at explicit boundaries.
It uses exported 16-bit ROM coefficients, including four-level precomputed
bilinear factors. `mamba_full_top.sv` implements the same operator order.

At publication the project must record the separately verified RTL numerical
status. Until both simulators pass `full-iverilog` and `full-verilator`, these
figures remain **software integer inference**, not full RTL inference results.

## No claim of chip implementation

The RTL is a functional microarchitecture candidate. It has not undergone
place-and-route on a selected FPGA or demonstrated timing closure, energy,
per-device LUT/DSP/BRAM use, or board-based classification.
