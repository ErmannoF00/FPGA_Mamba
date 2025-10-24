// ============================
// File: conv1d_shift.sv (REPLACE OLD FILE)
// Description: Causal conv1d for vector inputs (dim-wise)
// ============================


module conv1d_shift #(
parameter WIDTH = 16,
parameter DIM = 16,
parameter KERNEL_SIZE = 3
)(
input logic clk,
input logic rst_n,
input logic valid_in,
input logic [WIDTH-1:0] x_in [0:DIM-1],
output logic [WIDTH-1:0] y_out [0:DIM-1],
output logic valid_out
);


logic [WIDTH-1:0] kernel [0:DIM-1][0:KERNEL_SIZE-1];
logic [WIDTH-1:0] window [0:DIM-1][0:KERNEL_SIZE-1];
logic signed [2*WIDTH-1:0] acc;


initial begin
for (int d = 0; d < DIM; d++) begin
kernel[d][0] = 16'sd16384; // 0.5
kernel[d][1] = 16'sd8192; // 0.25
kernel[d][2] = 16'sd4096; // 0.125
end
end


always_ff @(posedge clk or negedge rst_n) begin
if (!rst_n) begin
for (int d = 0; d < DIM; d++) begin
for (int k = 0; k < KERNEL_SIZE; k++) begin
window[d][k] <= 0;
end
y_out[d] <= 0;
end
valid_out <= 0;
end else if (valid_in) begin
for (int d = 0; d < DIM; d++) begin
// Shift window
for (int k = KERNEL_SIZE-1; k > 0; k--) begin
window[d][k] <= window[d][k-1];
end
window[d][0] <= x_in[d];


acc = 0;
for (int k = 0; k < KERNEL_SIZE; k++) begin
acc += window[d][k] * kernel[d][k];
end
y_out[d] <= acc[WIDTH +: WIDTH]; // Normalize
end
valid_out <= 1;
end else begin
valid_out <= 0;
end
end


endmodule