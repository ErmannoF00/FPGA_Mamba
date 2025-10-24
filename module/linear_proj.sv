// ======================
// Description: Linear projection (matrix-vector multiplication)
// Purpose: Used for in_proj and out_proj layers in Mamba
// ======================

module linear_proj #(
    parameter IN_DIM    = 16,
    parameter OUT_DIM   = 16,
    parameter WIDTH     = 16,    // Q1.15 fixed-point
    parameter FRAC_BITS = 15
)(
    input  logic                        clk,
    input  logic                        rst_n,
    input  logic                        valid_in,
    input  logic signed [WIDTH-1:0]     x [0:IN_DIM-1], 
    output logic                        valid_out,
    output logic signed [WIDTH-1:0]     y [0:OUT_DIM-1] 
);

    // Weight matrix and bias
    logic signed [WIDTH-1:0] weight [0:OUT_DIM-1][0:IN_DIM-1];
    logic signed [WIDTH-1:0] bias   [0:OUT_DIM-1];

    // Accumulator: allow overflow bits
    logic signed [2*WIDTH-1:0] acc   [0:OUT_DIM-1];

    integer i, j;

    // ------------------------------
    // Load weights and biases from file
    // ------------------------------
    initial begin
        $readmemb("weights/in_proj_weight.mem", weight);
        $readmemb("weights/in_proj_bias.mem",   bias);
    end

    // ------------------------------
    // Matrix-vector multiply
    // ------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < OUT_DIM; i++) begin
                y[i]        <= '0;
                acc[i]      <= '0;
            end
            valid_out <= 0;
        end else begin
            if (valid_in) begin
                for (i = 0; i < OUT_DIM; i++) begin
                    acc[i] = 0;
                    for (j = 0; j < IN_DIM; j++) begin
                        acc[i] += x[j] * weight[i][j]; // 16b * 16b = 32b
                    end
                    acc[i] += bias[i] <<< FRAC_BITS; // align bias to match fixed-point scale
                    y[i] <= acc[i][FRAC_BITS +: WIDTH]; // extract middle bits (rounding optional)
                end
                valid_out <= 1;
            end else begin
                valid_out <= 0;
            end
        end
    end

endmodule
