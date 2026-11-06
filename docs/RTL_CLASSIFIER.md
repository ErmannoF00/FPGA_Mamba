# Classification from RTL scan outputs

The integration test replaces the recurrent-scan output tensor in a trained PyTorch model with values emitted by SystemVerilog simulation. It does not replace the projections, convolution, gate or final classifier with RTL.

## Data format

Fifty held-out digit images produce 400 scan transactions (eight row tokens per image). The vector exporter supplies Q6.10 `u`, `alpha`, `excitation`, `c` and `skip` buses to `precision_scan.sv`. The testbench compares each cycle with the independent integer model.

A consumed output transfer occurs only when `out_valid && out_ready`. With `+TRACE=<path>`, the testbench writes one 128-bit hexadecimal word for each such transfer. A complete 50-image trace contains 400 lines. Channel zero occupies the least significant 16 bits.

`model.rtl_classifier` parses the trace into signed integer values with shape `(50, 8, 8)`. It compares each element with the integer software reference and supplies the actual RTL values to the trained PyTorch output path. It then compares logits and predictions with the integer reference and original model, saving per-image results.

## Run

```bash
make -f Makefile.research scan-rtl-classify-iverilog
make -f Makefile.research scan-rtl-classify-verilator
make -f Makefile.research release-verify

cat build/rtl_classifier_iverilog.json
cat build/rtl_classifier_verilator.json
```

The release verifier rejects missing outputs, malformed hexadecimal words, inconsistent per-image predictions, incorrect checksums and mismatches between simulators.

## Hardware boundary

This result proves equivalence of the tested recurrent scan and its integer reference when driving a host classifier. It is not an FPGA implementation of the entire neural network. Complete on-device inference additionally needs parameter storage and transfer, projections, convolution, nonlinearities, the remaining output path and a device-specific top-level integration.
