// ===============================
// Description: Top-level Mamba inference wrapper for FPGA
// - Synthesizable (no $fwrite, no initial)
// - Inputs over switches or GPIO
// - Output on LEDs or external pin
// - ROM-based weight and LUT loading
// ===============================

module fpga_top #(
    parameter WIDTH     = 16,
    parameter IN_DIM    = 784,
    parameter OUT_DIM   = 10,
    parameter LUT_BITS  = 8
)(
    input  logic                 clk,
    input  logic                 rst_n,
    input  logic                 start,
    input  logic [WIDTH-1:0]    pixel_data,
    input  logic                 pixel_valid,
    output logic [3:0]          predicted_class,
    output logic                valid_out
);

    // ---------------------------
    // Input buffer (784 pixels)
    // ---------------------------
    logic [WIDTH-1:0] x_buf [0:IN_DIM-1];
    logic [$clog2(IN_DIM)-1:0] count;
    logic valid_ready;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count <= 0;
            valid_ready <= 0;
        end else if (start) begin
            count <= 0;
            valid_ready <= 0;
        end else if (pixel_valid && count < IN_DIM) begin
            x_buf[count] <= pixel_data;
            count <= count + 1;
            if (count == IN_DIM - 1)
                valid_ready <= 1;
        end
    end

    // ---------------------------
    // Core Mamba inference
    // ---------------------------
    logic [WIDTH-1:0] mamba_out [0:OUT_DIM-1];
    logic             mamba_valid;

    mamba_wrapper #(
        .WIDTH(WIDTH),
        .IN_DIM(IN_DIM),
        .OUT_DIM(OUT_DIM)
    ) core (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_ready),
        .x(x_buf),
        .valid_out(mamba_valid),
        .y(mamba_out)
    );

    // ---------------------------
    // Argmax (class prediction)
    // ---------------------------
    argmax #(
        .WIDTH(WIDTH),
        .N(OUT_DIM)
    ) top1 (
        .clk(clk),
        .valid_in(mamba_valid),
        .in_vec(mamba_out),
        .out_class(predicted_class),
        .valid_out(valid_out)
    );

endmodule
