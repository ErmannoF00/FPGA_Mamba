# Verification

The project maintains two independent RTL modules. Both are checked using software reference models and two event-based HDL simulators.

## Trained-model scan

The trained bilinear checkpoint generates 400 recurrent-scan transactions from 50 held-out digit images. A deterministic stimulus includes stalls, bubbles, resets and sequence boundaries. The testbench compares every cycle against the integer reference and writes only consumed output transfers to the trace.

Run:

```bash
make -f Makefile.research release-check
```

This command executes Python unit tests, Verilator lint, Icarus and Verilator RTL simulation, two RTL-fed classifier runs, trace/report consistency checks, and Yosys synthesis. It writes evidence to `build/rtl_classifier_iverilog.json`, `build/rtl_classifier_verilator.json`, and the matching `precision_scan_outputs_*.txt` files.

The release verifier checks that the simulators produce identical 400-output traces, that each trace is complete, and that the resulting 50 labels/predictions and checkpoint hashes match the reference.

## Scalar scan

`rtl/selective_scan.sv` uses signed Q4.12 with signed 16-bit input, state and output. Intermediate products are signed and extended before accumulation; a 12-bit arithmetic right shift implements floor rescaling on negative values, followed by saturation. State changes only upon `in_valid && in_ready`.

```bash
make reference lint directed verilator iverilog
make verilator LANES=8 SEEDS='1731 98989'
```

The four-lane verification uses three seeds with 2,400 oracle cycles each. The user's subsequent WSL test run reported 7,200 passing comparisons per simulator for four lanes. Eight-lane checks are available separately and should be rerun for release claims about that configuration.

## Recorded environment and scope

The November 2026 WSL run reported successful `release-check` and legacy checks. Yosys 0.33 reported zero structural problems and 159,012 generic cells for `precision_scan`. These results establish RTL simulation and generic synthesis for the tested configuration. They do not establish a mapped FPGA resource count, frequency, power or complete hardware inference.

The workflow `.github/workflows/ci.yml` covers both the trained scan and the scalar scan on a clean runner. A successful local run does not substitute for a confirmed successful GitHub Actions run after pushing.
