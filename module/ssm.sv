// =============================
// Description: Mamba-style SSM block with A, B, C, D loaded from .mem files
// Format: Q1.15 fixed-point
// =============================

module ssm #(
    parameter WIDTH = 16,
    parameter FRAC  = 15,
    parameter DEPTH = 1
)(
    input  logic              clk,
    input  logic              rst_n,
    input  logic              valid_in,
    input  logic [WIDTH-1:0]  x_in [0:DEPTH-1],
    output logic              valid_out,
    output logic [WIDTH-1:0]  y_out [0:DEPTH-1]
);

    // SSM weights
    logic signed [WIDTH-1:0] A [0:DEPTH-1];
    logic signed [WIDTH-1:0] B [0:DEPTH-1];
    logic signed [WIDTH-1:0] C [0:DEPTH-1];
    logic signed [WIDTH-1:0] D [0:DEPTH-1];

    // Internal state per neuron
    logic signed [WIDTH-1:0] h [0:DEPTH-1];
    logic signed [2*WIDTH-1:0] Ah, Bx, Ch, Dx;

    initial begin
        $readmemb("weights/ssm_A.mem", A);
        $readmemb("weights/ssm_B.mem", B);
        $readmemb("weights/ssm_C.mem", C);
        $readmemb("weights/ssm_D.mem", D);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < DEPTH; i++) begin
                h[i]      <= 0;
                y_out[i]  <= 0;
            end
            valid_out <= 0;
        end else if (valid_in) begin
            for (int i = 0; i < DEPTH; i++) begin
                Ah = A[i] * h[i];
                Bx = B[i] * x_in[i];
                h[i] <= (Ah + Bx) >>> FRAC;

                Ch = C[i] * h[i];
                Dx = D[i] * x_in[i];
                y_out[i] <= (Ch + Dx) >>> FRAC;
            end
            valid_out <= 1;
        end else begin
            valid_out <= 0;
        end
    end

endmodule
