// =========================
// File: mamba_block.sv
// Description: Full Mamba Block (in_proj → conv → GELU → ssm → out_proj + residual)
// Parameters: DIM = internal vector size
// =========================

module mamba_block #(parameter WIDTH=16, DIM=16)(
    input  logic                   clk,
    input  logic                   rst_n,
    input  logic                   valid_in,
    input  logic [WIDTH-1:0]      x_in [0:DIM-1],
    output logic                  valid_out,
    output logic [WIDTH-1:0]      x_out[0:DIM-1]
);

    logic        valid_proj, valid_conv, valid_gelu, valid_ssm;
    logic [WIDTH-1:0] x_proj    [0:DIM-1];
    logic [WIDTH-1:0] x_conv    [0:DIM-1];
    logic [WIDTH-1:0] x_gelu    [0:DIM-1];
    logic [WIDTH-1:0] x_ssm     [0:DIM-1];
    logic [WIDTH-1:0] x_outproj [0:DIM-1];

    // ------------------------
    // in_proj: Linear Layer
    // ------------------------
    linear_proj #(.IN_DIM(DIM), .OUT_DIM(DIM)) in_proj (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .x(x_in),
        .valid_out(valid_proj),
        .y(x_proj)
    );

    // ------------------------
    // conv1d_shift: Depthwise Conv
    // ------------------------
    conv1d_shift #(.WIDTH(WIDTH), .DIM(DIM)) conv (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_proj),
        .x_in(x_proj),
        .y_out(x_conv),
        .valid_out(valid_conv)
    );

    // ------------------------
    // GELU activation
    // ------------------------
    gelu_vector #(.WIDTH(WIDTH), .DIM(DIM)) gelu (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_conv),
        .x(x_conv),
        .y(x_gelu),
        .valid_out(valid_gelu)
    );

    // ------------------------
    // SSM block
    // ------------------------
    ssm_block #(.WIDTH(WIDTH), .DIM(DIM)) ssm (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_gelu),
        .x(x_gelu),
        .y(x_ssm),
        .valid_out(valid_ssm)
    );

    // ------------------------
    // out_proj
    // ------------------------
    linear_proj #(.IN_DIM(DIM), .OUT_DIM(DIM)) out_proj (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_ssm),
        .x(x_ssm),
        .valid_out(valid_out),
        .y(x_outproj)
    );

    // ------------------------
    // Residual connection
    // ------------------------
    genvar i;
    generate
        for (i = 0; i < DIM; i++) begin
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) x_out[i] <= 0;
                else        x_out[i] <= x_outproj[i] + x_in[i];
            end
        end
    endgenerate

endmodule
