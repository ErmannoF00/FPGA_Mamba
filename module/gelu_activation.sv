// ================================
// File: gelu_activation.sv
// Description: GELU(x) activation using tanh approximation
// Format: Q1.15 fixed-point input/output
// ================================

module gelu_activation #(
    parameter WIDTH = 16,
    parameter FRAC  = 15
)(
    input  logic signed [WIDTH-1:0] in_data,
    output logic signed [WIDTH-1:0] out_data
);

    // GELU approximation:
    // gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    logic signed [2*WIDTH-1:0] x2, x3, term1, tanh_arg, tanh_out;
    logic signed [WIDTH-1:0] gelu_mult, one_plus_tanh;

    always_comb begin
        x2 = in_data * in_data;
        x3 = (x2 >>> FRAC) * in_data; // x^3
        
        // term1 = x + 0.044715 * x^3
        // 0.044715 in Q1.15 ≈ 1465
        term1 = in_data <<< FRAC + ((x3 >>> FRAC) * 1465); // Q2.30
        
        // sqrt(2/pi) ≈ 0.79788 ≈ 26139 in Q1.15
        tanh_arg = (term1 >>> FRAC) * 26139; // Q2.30

        // Use tanh_arg[30:15] as LUT input (not implemented here)
        // Instead, we approximate tanh(x) ≈ x for small x
        tanh_out = tanh_arg >>> FRAC;

        // Final GELU = 0.5 * x * (1 + tanh(...))
        one_plus_tanh = ((1 <<< FRAC) + tanh_out[WIDTH-1:0]) >>> 1;
        gelu_mult = (in_data * one_plus_tanh) >>> FRAC;
        out_data = gelu_mult;
    end

endmodule
