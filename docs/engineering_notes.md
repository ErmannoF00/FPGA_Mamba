# Engineering notes

## Original repository problems

- `rms_norm.sv` has an unclosed loop/block and cannot be compiled.
- `mamba_block.sv` instantiates `ssm_block`, whereas the file defines `ssm`.
- `tb_mamba_wrapper.sv` instantiates undefined `mamba_top`.
- Makefile references `tb_mnist_predict.sv`; the existing file is `tb_mnist_pred.sv`.
- Shared `linear_proj` always reads the same `in_proj` weight paths for every instance.
- The only Python weights exported were classifier-head weights; RTL expects additional files.
- Python selective scan uses input-dependent discretization; RTL used fixed A/B/C/D.
- Convolution uses prior window contents and discards the newest token in the same cycle.
- Residual bypass is not latency-aligned with the pipelined path.
- The `gelu_vector` valid signal is clocked while its output is combinational.
- No evidence of synthesis, implementation on a physical FPGA, or software/RTL equivalence.

## New design limits

This project is a *verified building block* rather than a complete application.
Arithmetic uses signed Q4.12, floor-on-shift for negative numbers, saturating
16-bit state/output, and no overflow in 34-bit intermediate accumulators for
16-bit operands. Upstream must ensure coefficients have a useful numerical
range. The state register is updated only on successful input transfer.

## Verification gates

1. Python reference tests pass.
2. Verilator runs three seeds x 2,400 oracle cycles; an 8-lane configuration is checked separately.
3. Yosys accepts the RTL and reports mapped resource utilization.
4. Target-specific STA, area, power and timing closure are separate steps.

The uploaded environment had no Icarus Verilog/Verilator/Yosys installed; an
unrun hardware test is **not** recorded as passing. The Make targets expose
this dependency rather than silently skipping the check.

## Further integration work

Add parameterized state dimension >1, discretization from a specific Mamba
checkpoint, block-level streaming buffers, an actual memory subsystem and
platform integration. Establish PyTorch/fixed-point/RTL agreement on recorded
token sequences before reporting end-to-end accuracy or FPGA performance.
