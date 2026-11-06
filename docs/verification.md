# Verification specification

## Numeric behavior

- Signed input/parameter/state/output width: 16 bits, two's-complement.
- Fractional bits: 12. Default `LANES=4`, `D_Q=2048`.
- Signed multiplication: 16 × 16 → 32 bits; two 32-bit products are sign-extended to 34 bits before addition.
- Rescale: arithmetic right shift by 12 (floor for negative fractions).
- Saturate after the complete sum; no independent product saturation.
- Coefficients vary on every accepted token. `sequence_start` substitutes zero for *previous* state before the recurrence.

## Sequential behavior

- Rising-edge input acceptance requires `in_valid && in_ready`.
- `in_ready = !out_valid || out_ready`.
- State updates only on accepted input; a stalled output cannot be overwritten.
- `out_valid` may deassert on a cycle with no accepted input after an output was consumed.
- `rst_n` asynchronously clears the state and the entire output register.
- All output bits are checked even when `out_valid=0` to verify register-hold behavior, though downstream logic must not consume invalid output data.

## Required local gates

| Gate | Command | Acceptance criterion |
|---|---|---|
| Reference tests | `make reference` | All integer/math/protocol tests pass |
| RTL lint | `make lint` | Zero Verilator warnings and errors |
| Directed arithmetic | `make directed` | Hand-computed results for positive/negative cases, stalls, reset and saturation |
| RTL equivalence (4 lanes) | `make verilator` | Three seeds × 2,400 cycles pass |
| Independent simulator | `make iverilog` | Same reference vectors pass Icarus |
| RTL equivalence (8 lanes) | `make verilator LANES=8 SEEDS='1731 98989'` | Both seeds pass |
| Generic synthesis | `make synth` | Yosys completes with `check -assert` |
| FPGA place and route | `make vivado PART=...` | Reports identify exact device, tool version, timing constraints, WNS and resources |

The randomized testbench samples control and registered outputs one nanosecond after the falling-edge stimulus assignment, before the next rising edge. The Python model returns that pre-edge observation, then performs the rising-edge state transition for its next vector row. Asynchronous reset is exercised during active traffic.

## Recorded WSL results before the correction

- Python: 7/7 unit tests passed.
- Icarus: 3 × 2,400 cycle comparisons passed, `LANES=4`.
- Verilator: failed at cycle 4 in both 4-lane and 8-lane configurations; all output lanes reported `0x7fff` for one non-saturating vector.
- Verilator warnings: truncation of the integer `D_Q` parameter; width expansion in signed saturation thresholds; missing module timescale.
- Yosys 0.33: generic synthesis completed, 24,260 generic cells **for that pre-correction RTL**.

These numbers must not be presented as successful verification of the current RTL. Re-run and save logs from the corrected source before creating a commit.

## Remaining system-level tests

1. Compare a multidimensional selective scan against a PyTorch/Mamba reference, including coefficient discretization and signed quantization tolerances.
2. Check input/output projection and convolution individually, including tensor layout, boundary padding and weight packing.
3. Add whole-block inference with at least one fixed trained checkpoint, comparing per-layer activations and final model predictions.
4. Measure RTL throughput as accepted/retired transactions per cycle and backpressure distribution with a controlled driver.
5. Implement a clock-constrained design for a specified FPGA; report LUTs, FFs, DSPs, BRAM, Fmax/WNS and power methodology.
6. Validate clock-domain crossing, host transfer, SRAM interface, and actual FPGA board operation before reporting hardware inference.
