// ================================
// File: gelu_vector.sv
// Description: GELU applied elementwise to vector input
// ================================

module gelu_vector #(
    parameter WIDTH = 16,
    parameter DIM   = 16
)(
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     valid_in,
    input  logic signed [WIDTH-1:0] x [0:DIM-1],
    output logic signed [WIDTH-1:0] y [0:DIM-1],
    output logic                     valid_out
);

    genvar i;
    generate
        for (i = 0; i < DIM; i++) begin : gelu_block
            gelu_activation gelu_inst (
                .in_data(x[i]),
                .out_data(y[i])
            );
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_out <= 0;
        else        valid_out <= valid_in;
    end

endmodule
