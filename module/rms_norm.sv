// ============================================================
// File: rms_norm.sv
// Description: RMS Normalization for vector input in Q1.15
// ------------------------------------------------------------
// Formula: y[i] = x[i] * (1 / sqrt(mean(x[i]^2) + ε))
// Uses precomputed LUT (rsqrt_lut_q15.mem)
// ============================================================

module rms_norm #(
    parameter WIDTH     = 16,     // Q1.15 format
    parameter IN_DIM    = 784,    // number of elements (MNIST image)
    parameter LUT_BITS  = 8,      // 256-entry LUT
    parameter FRAC_BITS = 15
)(
    input  logic                        clk,
    input  logic                        rst_n,
    input  logic                        valid_in,
    input  logic signed [WIDTH-1:0]     x [0:IN_DIM-1],
    output logic                        valid_out,
    output logic signed [WIDTH-1:0]     y [0:IN_DIM-1]
);

    // ------------------------------
    // Internal memories and signals
    // ------------------------------
    logic [WIDTH-1:0] rsqrt_lut [0:(1<<LUT_BITS)-1];
    logic signed [2*WIDTH-1:0] square [0:IN_DIM-1];
    logic signed [2*WIDTH+15:0] sum_squares;
    logic signed [WIDTH-1:0] mean_q15;
    logic [WIDTH-1:0] rsqrt_val;
    logic signed [2*WIDTH-1:0] mult;
    integer i;

    // ------------------------------
    // Load reciprocal sqrt LUT
    // ------------------------------
    initial begin
        $readmemb("data/rsqrt_lut_q15.mem", rsqrt_lut);
    end

    // ------------------------------
    // Compute RMS normalization
    // ------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out   <= 0;
            sum_squares <= 0;
            for (i = 0; i < IN_DIM; i++) begin
                square[i] <= 0;
                y[i]      <= 0;
            end
        end else begin
            if (valid_in) begin
                // 1) Compute sum of squares
                sum_squares = 0;
                for (i = 0; i < IN_DIM; i++) begin
                    square[i] = x[i] * x[i];  // Q2.30
                    sum_squares += square[i];
                end

                // 2) Mean of squares = sum / IN_DIM
                mean_q15 = (sum_squares >>> FRAC_BITS) / IN_DIM;
                // 3) Get rsqrt(mean) from LUT
                // Map mean [0,1] → [0,255]
                rsqrt_val = rsqrt_lut[ mean_q15[15:8] ];

                // 4) Normalize each element
                for (i = 0; i < IN_DIM; i++) begin
                    mult  = x[i] * rsqrt_val;
                    y[i]  <= mult[FRAC_BITS +: WIDTH]; // scale back to Q1.15

                valid_out <= 1;
            end else begin
                valid_out <= 0;
            end
        end
    end

endmodule
