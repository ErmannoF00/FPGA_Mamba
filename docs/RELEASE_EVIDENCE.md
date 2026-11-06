# Recorded verification and measurements

## Local WSL verification, November 2026

| Test | Recorded outcome |
| :--- | :--- |
| Python tests | 18 passed |
| Trained scan, Icarus | 712 cycles passed |
| Trained scan, Verilator | 712 cycles passed |
| RTL-consumed recurrent output trace | 400 outputs |
| RTL-fed digit classification | 49 correct out of 50, Icarus report |
| RTL versus integer software | 0 maximum logit error; no prediction changes |
| RTL versus floating-point model | 1 different prediction on the 50-sample subset |
| Four-lane scalar scan, Icarus | 3 seeds × 2,400 cycles passed |
| Four-lane scalar scan, Verilator | 3 seeds × 2,400 cycles passed |
| Research `release-check` | Exit code 0 |
| Yosys 0.33 `precision_scan` | Zero structural problems, 159,012 generic mapped cells |

`release-check` includes both simulator paths and a verifier that compares their traces and predictions. The two standalone classification report files are generated locally in `build/` and are not checked into source control.

The digits model has 96.11% bilinear and 95.56% ZOH accuracy over 360 held-out test images. The 98% RTL-fed accuracy is measured on a 50-image subset. Comparing those percentages directly without accounting for sample size would be misleading.

## State-width experiment

The calibrated 32-state allocation uses 256 nominal register bits, compared with 512 bits for 32 full-precision signed 16-bit states. On the recorded test set, the logit mean-squared error for the calibrated allocation was 0.3110 versus 0.3893 for a uniform equal-budget allocation. These are software numerical results, not FPGA area or energy measurements.

## Boundaries

The RTL implements only the recurrent scan. Parameter generation, input/output projections, depthwise convolution, gate, residual path and classification head run in PyTorch. `precision_config.svh` is generated offline; no runtime precision controller is implemented. No FPGA synthesis target, placement, timing, device measurements or hardware energy estimates are supplied.

These logs document the tested behavior without establishing performance improvements against published FPGA Mamba accelerators.
