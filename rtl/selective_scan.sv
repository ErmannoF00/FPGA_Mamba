`timescale 1ns / 1ps

// Signed fixed-point diagonal state-space recurrence.
// Each accepted input token updates one state and one output per lane.
// Products are evaluated at full precision; saturation follows Q rescaling.
module selective_scan #(
    parameter integer WIDTH = 16,
    parameter integer FRAC  = 12,
    parameter integer LANES = 4,
    parameter signed [WIDTH-1:0] D_Q = 16'sd2048
) (
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          in_valid,
    output wire                          in_ready,
    input  wire                          sequence_start,
    input  wire signed [LANES*WIDTH-1:0] x,
    input  wire signed [LANES*WIDTH-1:0] a,
    input  wire signed [LANES*WIDTH-1:0] b,
    input  wire signed [LANES*WIDTH-1:0] c,
    output reg                           out_valid,
    input  wire                          out_ready,
    output reg signed [LANES*WIDTH-1:0] y
);
    localparam integer PRODUCT_WIDTH = 2 * WIDTH;
    localparam integer SUM_WIDTH     = PRODUCT_WIDTH + 2;

    localparam signed [WIDTH-1:0] MAX_VALUE = {1'b0, {(WIDTH-1){1'b1}}};
    localparam signed [WIDTH-1:0] MIN_VALUE = {1'b1, {(WIDTH-1){1'b0}}};
    localparam signed [SUM_WIDTH-1:0] MAX_EXT =
        {{(SUM_WIDTH-WIDTH){1'b0}}, MAX_VALUE};
    localparam signed [SUM_WIDTH-1:0] MIN_EXT =
        {{(SUM_WIDTH-WIDTH){1'b1}}, MIN_VALUE};

    reg  signed [WIDTH-1:0] state [0:LANES-1];
    wire signed [WIDTH-1:0] x_lane [0:LANES-1];
    wire signed [WIDTH-1:0] a_lane [0:LANES-1];
    wire signed [WIDTH-1:0] b_lane [0:LANES-1];
    wire signed [WIDTH-1:0] c_lane [0:LANES-1];
    wire signed [WIDTH-1:0] prev_state [0:LANES-1];
    wire signed [WIDTH-1:0] next_state [0:LANES-1];
    wire signed [WIDTH-1:0] next_output [0:LANES-1];

    wire signed [PRODUCT_WIDTH-1:0] state_product [0:LANES-1];
    wire signed [PRODUCT_WIDTH-1:0] input_product [0:LANES-1];
    wire signed [PRODUCT_WIDTH-1:0] output_product [0:LANES-1];
    wire signed [PRODUCT_WIDTH-1:0] skip_product [0:LANES-1];
    wire signed [SUM_WIDTH-1:0] state_sum [0:LANES-1];
    wire signed [SUM_WIDTH-1:0] output_sum [0:LANES-1];

    // The arithmetic shift rounds negative fractions toward -infinity.
    // Both comparator operands have the same explicitly signed width.
    function automatic signed [WIDTH-1:0] sat_q;
        input signed [SUM_WIDTH-1:0] value;
        reg signed [SUM_WIDTH-1:0] scaled;
        begin
            scaled = value >>> FRAC;
            if (scaled > MAX_EXT)
                sat_q = MAX_VALUE;
            else if (scaled < MIN_EXT)
                sat_q = MIN_VALUE;
            else
                sat_q = scaled[WIDTH-1:0];
        end
    endfunction

    // A one-entry output register permits simultaneous dequeue/enqueue.
    assign in_ready = !out_valid || out_ready;

    genvar lane;
    generate
        for (lane = 0; lane < LANES; lane = lane + 1) begin : datapath
            assign x_lane[lane] = x[lane*WIDTH +: WIDTH];
            assign a_lane[lane] = a[lane*WIDTH +: WIDTH];
            assign b_lane[lane] = b[lane*WIDTH +: WIDTH];
            assign c_lane[lane] = c[lane*WIDTH +: WIDTH];
            assign prev_state[lane] = sequence_start ? {WIDTH{1'b0}} : state[lane];

            // Full-width signed multiplication, followed by explicit
            // sign extension before summing. No implicit expression sizing.
            assign state_product[lane]  = $signed(a_lane[lane]) * $signed(prev_state[lane]);
            assign input_product[lane]  = $signed(b_lane[lane]) * $signed(x_lane[lane]);
            assign state_sum[lane] =
                $signed({{2{state_product[lane][PRODUCT_WIDTH-1]}}, state_product[lane]}) +
                $signed({{2{input_product[lane][PRODUCT_WIDTH-1]}}, input_product[lane]});
            assign next_state[lane] = sat_q(state_sum[lane]);

            assign output_product[lane] = $signed(c_lane[lane]) * $signed(next_state[lane]);
            assign skip_product[lane]   = $signed(D_Q) * $signed(x_lane[lane]);
            assign output_sum[lane] =
                $signed({{2{output_product[lane][PRODUCT_WIDTH-1]}}, output_product[lane]}) +
                $signed({{2{skip_product[lane][PRODUCT_WIDTH-1]}}, skip_product[lane]});
            assign next_output[lane] = sat_q(output_sum[lane]);
        end
    endgenerate

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            y <= {LANES*WIDTH{1'b0}};
            for (i = 0; i < LANES; i = i + 1)
                state[i] <= {WIDTH{1'b0}};
        end else if (in_ready) begin
            out_valid <= in_valid;
            if (in_valid) begin
                for (i = 0; i < LANES; i = i + 1) begin
                    state[i] <= next_state[i];
                    y[i*WIDTH +: WIDTH] <= next_output[i];
                end
            end
        end
    end
endmodule
