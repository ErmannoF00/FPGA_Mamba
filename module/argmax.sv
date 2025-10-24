// ======================
// File: argmax.sv
// Description: Picks argmax from a 10-class output vector
// ======================

module argmax #(
    parameter WIDTH = 16,
    parameter N = 10
)(
    input  logic signed [WIDTH-1:0] in_vec [0:N-1],
    input  logic                    valid_in,
    input  logic                    clk,
    output logic        [3:0]       out_class, 
    output logic                    valid_out
);

    always_ff @(posedge clk) begin
        if (valid_in) begin
            integer i;
            logic signed [WIDTH-1:0] max_val;
            logic        [3:0]       max_idx;
            max_val = in_vec[0];
            max_idx = 0;
            for (i = 1; i < N; i++) begin
                if (in_vec[i] > max_val) begin
                    max_val = in_vec[i];
                    max_idx = i;
                end
            end
            out_class <= max_idx;
            valid_out <= 1;
        end else begin
            valid_out <= 0;
        end
    end

endmodule
