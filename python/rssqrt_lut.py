import numpy as np

# -------------------------
# LUT CONFIGURATION
# -------------------------
LUT_BITS = 8           # 256 entries
Q = 15                 # Q1.15 format
XMIN = 1 / (1 << LUT_BITS) 
XMAX = 1.0
ENTRIES = 1 << LUT_BITS
SCALE = 1 << Q         

# -------------------------
# LUT COMPUTATION
# -------------------------
x_vals = np.linspace(XMIN, XMAX, ENTRIES)
rsqrt_vals = 1 / np.sqrt(x_vals)
rsqrt_q15 = np.clip((rsqrt_vals * SCALE).round(), 0, 65535).astype(int)

# -------------------------
# Save as .mem format (for $readmemh or $readmemb)
# -------------------------
with open("data/rsqrt_lut_q15.mem", "w") as f:
    for val in rsqrt_q15:
        f.write(f"{val:016b}\n")  # write as binary
print("Saved to rsqrt_lut_q15.mem")

# -------------------------
# Also print for SystemVerilog case/init block
# -------------------------
print("initial begin")
for i, val in enumerate(rsqrt_q15):
    print(f"  rsqrt_lut[{i}] = 16'd{val};")
print("end")
