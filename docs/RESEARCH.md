# Recurrent state precision allocation

## Question

Does per-state fractional precision selected from held-out calibration sensitivity reduce output distortion compared with a uniform state format under an identical state-bit budget?

## Model and method

The classifier uses eight channels and four diagonal recurrent states per channel. The model supports bilinear or ZOH coefficient discretization. Its external scan values are signed Q6.10, while internal state fractional widths are configurable at synthesis time.

The calibration estimates the effect of quantizing individual states on network logits using the validation set. A pole-lifetime factor based on the maximum transition magnitude contributes a stability-related weight. A dynamic program minimizes the resulting additive proxy under an exact 64-fractional-bit budget. The selected map is stored in `results/precision64.json` and exported to `rtl/precision_config.svh`.

This method is heuristic and offline. It is not a certified error bound or a token-by-token adaptive-precision controller.

## Recorded test result

| State format | Logit MSE versus Q6.10 | Nominal state-storage bits |
| :--- | ---: | ---: |
| Uniform Q6.10 | 0 | 512 |
| Uniform two fractional bits | 0.3893 | 256 |
| Calibrated, 64 fractional bits total | 0.3110 | 256 |

The calibrated configuration reduces logit MSE by approximately 20.1% compared with the uniform equal-budget format on the recorded checkpoint and data split. It does not establish an accuracy improvement over the uniform configuration, a hardware speedup or a reduction in mapped FPGA area.

## Reproduce

```bash
make -f Makefile.research research-evaluate
make -f Makefile.research research-precision
make -f Makefile.research research-hybrid
make -f Makefile.research release-check
```

The saved map in `results/precision64.json` is the release reference. `research-precision` writes a fresh map to `build/precision64.json`; it does not silently replace the checked-in configuration.

## Further work

A runtime implementation would require mode-dependent arithmetic, a conservative propagated-error invariant, mode-switch control logic and measurable hardware benefits. None of these is present in the checked-in RTL. Before claiming novelty, compare against uniform and calibrated fixed precision, alternate sensitivity metrics, and published FPGA SSM/Mamba quantization methods on matched workloads and FPGA devices.
