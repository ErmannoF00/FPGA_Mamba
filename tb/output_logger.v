// ======================
// File: output_logger.v
// Description: Dumps output values to output.txt (for Python parsing)
// ======================

module output_logger #(parameter WIDTH = 16) (
    input logic clk,
    input logic valid,
    input logic signed [WIDTH-1:0] data [0:9]  // assuming 10-class output
);

    integer f;
    initial f = $fopen("output.txt", "w");

    always @(posedge clk) begin
        if (valid) begin
            $fwrite(f, "");
            for (int i = 0; i < 10; i++) begin
                $fwrite(f, "%0d\n", data[i]);
            end
        end
    end
endmodule
