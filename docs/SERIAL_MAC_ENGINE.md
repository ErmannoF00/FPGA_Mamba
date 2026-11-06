# Single-MAC inference microarchitecture (candidate until tested)

## Purpose and provenance

The original `rtl/mamba_full_top.sv` achieved exact software-oracle agreement on
360 held-out digit images in Icarus and Verilator, but generic Yosys synthesis
consumed approximately 6.5 GiB RAM before completion. Its nested combinational
projection loops describe multiple simultaneously instantiated signed multipliers.

`rtl/mamba_serial_top.sv` is an alternative implementation of the **same** trained
Q6.10 8-token, 8-channel, 4-state Mamba-derived classifier. It does not modify
`rtl/precision_scan.sv` or model ROM contents. It contains a single explicit
16x16 signed multiplier, reused by all non-scan layers. The selective scan
retains its own internal arithmetic. Do not misstate this as one multiplier in
**the entire design**.

## Schedule

| Phase | Work per token | Scheduled cycles, excluding handshakes |
| --- | --- | ---: |
| PROJ | Input + gate projections (8x8 each) | 128 |
| CONV | Three taps for each of eight channels | 24 |
| PARAM | B and C (4x8 each), delta (8x8) | 128 |
| EXCITE_A/B | Two sequential quantized multiplications per state | 64 |
| FEED/CONSUME | Verified precision scan transaction | variable |
| GATE | Gate activation and multiply | 8 |
| POST | Trained 8x8 output projection and residual | 64 |
| HEAD | Ten classes, 64 features per class | 640 per image |

Approximately 3,328 non-scan cycles for the eight tokens plus 640 head cycles,
excluding phase boundaries and scan handshakes. These are *design-derived
counts*, not measured cycles or FPGA throughput. The testbench permits 12,000
cycles per sample for a conservative functional timeout.

## Numeric behavior

- Signed Q6.10 16-bit operands, 32-bit products, 64-bit accumulators.
- Nearest-even shift by ten fractional bits; saturating signed 16-bit output.
- The convolution rounds and saturates **each tap product before summation**,
  matching the existing integer reference.
- The output projection saturates, then adds the projected residual and
  saturates again. These two saturation stages must not be combined.
- Quantized B/C/delta parameters and the 128-pair bilinear ROM are unchanged.
- The classifier tie behavior selects the lowest class index.
- One 16x16 multiplier is shared by the serial top's non-scan operations;
  memory indexing and narrow divide-by-six gating remain combinational.

## Reproducible checks (from repo root)

Run in order, stopping on any failure:

```bash
make -f Makefile.full full-export
make -f Makefile.serial serial-lint
make -f Makefile.serial serial-smoke
make -f Makefile.serial serial-iverilog
make -f Makefile.serial serial-verilator
make -f Makefile.serial serial-coarse
make -f Makefile.serial serial-synth
```

`serial-iverilog` and `serial-verilator` compare ten signed output logits for
**all 360 held-out images**, not only predicted labels. A passing Verilator
simulation is not a substitute for synthesis or place-and-route. The synthesis
rules include time limits; a timeout is a failure, not a successful result.

## Promotion / release criterion

Until all checks pass, leave `mamba_full_top` and the old testbench unchanged.
Once they pass, promote the serial top to the canonical inference entry point,
keep the original proven implementation as a non-default correctness reference,
and report generic resource utilization separately from FPGA post-route PPA.
