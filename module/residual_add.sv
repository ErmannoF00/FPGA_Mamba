
module residual_add #(
    parameter WIDTH = 16
)(
    input  logic [WIDTH-1:0] x,
    input  logic [WIDTH-1:0] y,
    output logic [WIDTH-1:0] out
);

    logic signed [WIDTH-1:0] x_s, y_s;
    logic signed [WIDTH:0]   sum;

    assign x_s = x;
    assign y_s = y;
    assign sum = x_s + y_s;
    assign out = sum[WIDTH-1:0];

endmodule
