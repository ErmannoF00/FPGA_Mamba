# Trained-model classification using RTL scan outputs

This integration stage feeds **actual outputs of the RTL recurrent scan** into the
trained output projection, gate and classification head implemented in PyTorch.
It does **not** implement these front-end/back-end operators in RTL. Therefore
it is a hardware-in-the-loop *simulation*, not FPGA inference.

## Data contract

The `model.export_scan_vectors` exporter builds the sequence of 400 recurrent
transactions (50 images × 8 row tokens) and a randomized ready/valid schedule.
The testbench compares the RTL buses cycle by cycle with the independent
integer reference. With `+TRACE=build/precision_scan_outputs.txt`, it writes
one 128-bit output per **consumed** output transaction, not per clock cycle.
The first hexadecimal word represents sample 0, token 0; the least significant
16 bits correspond to output channel 0. There must be exactly 400 lines.

The host parser converts every signed Q6.10 word back to an integer tensor
`(50, 8, 8)`, checks each element against the software recurrence, injects
the RTL-produced scan outputs into the actual trained PyTorch output projection
and classification head, and records every predicted digit. It also compares
the resulting logits against the integer reference and floating-point model.

## Commands

```bash
cd ~/projects/FPGA_Mamba
python3 -m unittest tests.test_rtl_classifier -v
make -f Makefile.research scan-rtl-classify-iverilog
make -f Makefile.research scan-rtl-classify-verilator
cat build/rtl_classifier.json
```

If the trace has fewer than 400 output handshakes, the parser fails; if one
signed channel disagrees with the reference, the classifier fails before
producing a misleading accuracy result. The report contains input-checkpoint
and RTL-trace hashes, per-image labels/predictions, and logit differences.

## Boundaries and next experiments

1. Replace the PyTorch input projection, convolution, selective-parameter
   generator, gate, output projection and classifier head with specified and
   independently tested hardware modules.
2. Treat the state-precision map as **compile-time configured**, not adaptive
   per token. Current `precision64.json` is a calibrated offline allocation.
3. Implement hardware-selectable widths or packed narrow datapaths before
   claiming dynamic precision or energy efficiency.
4. Evaluate resource occupancy, fMAX, latency and power on a named FPGA using
   vendor tools and, where possible, a physical board.
5. Compare uncertainty/error-bound controllers against precision allocation
   heuristics with matched memory, DSP and operating-frequency constraints.
