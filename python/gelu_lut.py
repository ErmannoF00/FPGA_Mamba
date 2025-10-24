# ===============================
# Description: Generate LUT for GELU(x) in Q1.15 fixed-point
# GELU(x) ≈ 0.5x(1 + tanh[√(2/π)(x + 0.0447x^3)])
# Input domain: [-8.0, +8.0] mapped to 256 entries
# ===============================

import numpy as np

ENTRIES = 256
X_MIN = -8.0
X_MAX = 8.0
Q = 15
SCALE = 1 << Q

x_vals = np.linspace(X_MIN, X_MAX, ENTRIES)
inner = np.sqrt(2/np.pi) * (x_vals + 0.044715 * x_vals**3)
gelu_vals = 0.5 * x_vals * (1 + np.tanh(inner))

# Convert to Q1.15
gelu_q15 = np.clip(np.round(gelu_vals * SCALE), -32768, 32767).astype(np.int16)

with open("data/gelu_lut_q15.mem", "w") as f:
    for val in gelu_q15:
        f.write(f"{val & 0xFFFF:016b}\n")

print("GELU LUT generated and saved to gelu_lut_q15.mem")

