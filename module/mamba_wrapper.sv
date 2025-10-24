// =============================
// File: mamba_wrapper.sv
// Description: Full multi-layer Mamba pipeline for MNIST FPGA inference
// =============================

module mamba_wrapper #(
    parameter WIDTH     = 16,
    parameter IN_DIM    = 784,
    parameter HIDDEN_DIM= 16,
    parameter OUT_DIM   = 10,
    parameter FRAC_BITS = 15,
    parameter N_LAYERS  = 2
)(
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     valid_in,
    input  logic [WIDTH-1:0]        x [0:IN_DIM-1],
    output logic                    valid_out,
    output logic [WIDTH-1:0]        y [0:OUT_DIM-1]
);

    logic        valid_embed, valid_final;
    logic [WIDTH-1:0] embed_out [0:HIDDEN_DIM-1];
    logic [WIDTH-1:0] mamba_out [0:HIDDEN_DIM-1];

    // --- Embedding (linear projection from input) ---
    linear_proj #( .IN_DIM(IN_DIM), .OUT_DIM(HIDDEN_DIM), .WIDTH(WIDTH) ) embed (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .x(x),
        .valid_out(valid_embed),
        .y(embed_out)
    );

    // --- Stack of Mamba Blocks ---
    logic [WIDTH-1:0] layer_in  [0:HIDDEN_DIM-1];
    logic [WIDTH-1:0] layer_out [0:HIDDEN_DIM-1];
    logic [N_LAYERS:0] valid_chain;
    assign layer_in = embed_out;
    assign valid_chain[0] = valid_embed;

    genvar i;
    generate
        for (i = 0; i < N_LAYERS; i++) begin : mamba_stack
            mamba_block #( .DIM(HIDDEN_DIM), .WIDTH(WIDTH) ) blk (
                .clk(clk),
                .rst_n(rst_n),
                .valid_in(valid_chain[i]),
                .x(layer_in),
                .valid_out(valid_chain[i+1]),
                .y(layer_out)
            );
            if (i < N_LAYERS - 1) begin
                assign layer_in = layer_out;
            end else begin
                assign mamba_out = layer_out;
                assign valid_final = valid_chain[N_LAYERS];
            end
        end
    endgenerate

    // --- Output Projection ---
    linear_proj #( .IN_DIM(HIDDEN_DIM), .OUT_DIM(OUT_DIM), .WIDTH(WIDTH) ) out_proj (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_final),
        .x(mamba_out),
        .valid_out(valid_out),
        .y(y)
    );

endmodule
