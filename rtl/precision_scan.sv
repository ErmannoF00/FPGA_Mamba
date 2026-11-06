`timescale 1ns / 1ps

// Parameterized recurrent-state bank with individually sized state registers.
// All external data use signed Q6.10; lane states use signed Q6.STATE_FRAC.
// Coefficients (alpha, drive, c) are supplied by the model front end.
// The input excitation is already the quantized product gain * B * u.
// This block is the recurrent SSM, not the entire neural-network classifier.
module precision_scan #(
    parameter integer WIDTH = 16,
    parameter integer FRAC = 10,
    parameter integer INT_BITS = 6,
    parameter integer CHANNELS = 8,
    parameter integer STATES = 4,
    parameter [CHANNELS*STATES*4-1:0] STATE_FRAC_MAP =
        {(CHANNELS*STATES){4'd10}}
) (
    input  wire clk,
    input  wire rst_n,
    input  wire in_valid,
    output wire in_ready,
    input  wire sequence_start,
    input  wire signed [CHANNELS*WIDTH-1:0] u,
    input  wire signed [CHANNELS*STATES*WIDTH-1:0] alpha,
    input  wire signed [CHANNELS*STATES*WIDTH-1:0] excitation,
    input  wire signed [STATES*WIDTH-1:0] c,
    input  wire signed [CHANNELS*WIDTH-1:0] skip,
    output reg out_valid,
    input  wire out_ready,
    output reg signed [CHANNELS*WIDTH-1:0] y
);
    assign in_ready = !out_valid || out_ready;
    // Integer nearest-even rescale for both positive and negative values.
    function automatic signed [63:0] round_shift;
        input signed [63:0] value;
        input integer shift;
        reg signed [63:0] floor_value, remainder, half;
        begin
            if (shift == 0) begin
                round_shift = value;
            end else begin
                floor_value = value >>> shift;
                remainder = value - (floor_value <<< shift);
                half = 64'sd1 <<< (shift - 1);
                if ((remainder > half) ||
                    ((remainder == half) && floor_value[0]))
                    round_shift = floor_value + 64'sd1;
                else
                    round_shift = floor_value;
            end
        end
    endfunction

    function automatic signed [WIDTH-1:0] clip_word;
        input signed [63:0] value;
        reg signed [63:0] max_word, min_word;
        begin
            max_word = (64'sd1 <<< (WIDTH-1)) - 64'sd1;
            min_word = -(64'sd1 <<< (WIDTH-1));
            if (value > max_word)
                clip_word = {1'b0,{(WIDTH-1){1'b1}}};
            else if (value < min_word)
                clip_word = {1'b1,{(WIDTH-1){1'b0}}};
            else
                clip_word = value[WIDTH-1:0];
        end
    endfunction

    wire signed [WIDTH-1:0] next_full [0:CHANNELS*STATES-1];
    wire signed [63:0] projected [0:CHANNELS*STATES-1];

    genvar d, n;
    generate
        for (d=0; d<CHANNELS; d=d+1) begin: channel
            for (n=0; n<STATES; n=n+1) begin: pole
                localparam integer INDEX = d*STATES+n;
                localparam integer FBITS = {28'b0, STATE_FRAC_MAP[4*INDEX +: 4]};
                localparam integer STATE_WIDTH = INT_BITS+FBITS;
                localparam signed [63:0] MAX_STATE = (64'sd1 <<< (STATE_WIDTH-1))-1;
                localparam signed [63:0] MIN_STATE = -(64'sd1 <<< (STATE_WIDTH-1));
                reg signed [STATE_WIDTH-1:0] state;
                wire signed [WIDTH-1:0] a_word;
                wire signed [WIDTH-1:0] e_word;
                wire signed [WIDTH-1:0] c_word;
                wire signed [STATE_WIDTH-1:0] prev_state;
                wire signed [WIDTH+STATE_WIDTH-1:0] state_product;
                wire signed [63:0] product_ext;
                wire signed [63:0] excitation_ext;
                wire signed [63:0] state_sum;
                wire signed [STATE_WIDTH-1:0] next_state;
                wire signed [2*WIDTH-1:0] output_product;
                wire signed [63:0] output_ext;

                assign a_word = alpha[INDEX*WIDTH +: WIDTH];
                assign e_word = excitation[INDEX*WIDTH +: WIDTH];
                assign c_word = c[n*WIDTH +: WIDTH];
                assign prev_state = sequence_start ? {STATE_WIDTH{1'b0}} : state;
                assign state_product = $signed(a_word)*$signed(prev_state);
                assign product_ext = {{(64-WIDTH-STATE_WIDTH){state_product[WIDTH+STATE_WIDTH-1]}},state_product};
                assign excitation_ext = {{(64-WIDTH){e_word[WIDTH-1]}},e_word};
                assign state_sum = round_shift(product_ext,FRAC)
                                 + round_shift(excitation_ext,FRAC-FBITS);
                assign next_state = (state_sum > MAX_STATE) ? MAX_STATE[STATE_WIDTH-1:0]
                                  : (state_sum < MIN_STATE) ? MIN_STATE[STATE_WIDTH-1:0]
                                  : state_sum[STATE_WIDTH-1:0];
                wire signed [WIDTH-1:0] next_extended;
                assign next_extended = {{(WIDTH-STATE_WIDTH){next_state[STATE_WIDTH-1]}},next_state};
                assign next_full[INDEX] = next_extended <<< (FRAC-FBITS);
                assign output_product = $signed(c_word)*$signed(next_full[INDEX]);
                assign output_ext = {{(64-2*WIDTH){output_product[2*WIDTH-1]}},output_product};
                assign projected[INDEX] = round_shift(output_ext,FRAC);

                always @(posedge clk or negedge rst_n)
                    if (!rst_n) state <= {STATE_WIDTH{1'b0}};
                    else if (in_ready && in_valid) state <= next_state;
            end
        end
    endgenerate

    integer i,j;
    reg signed [63:0] acc;
    reg signed [2*WIDTH-1:0] skip_product;
    reg signed [63:0] skip_ext;
    reg signed [CHANNELS*WIDTH-1:0] next_y;
    always @* begin
        next_y = {CHANNELS*WIDTH{1'b0}};
        acc = 64'sd0;
        skip_product = {(2*WIDTH){1'b0}};
        skip_ext = 64'sd0;
        for (i=0;i<CHANNELS;i=i+1) begin
            acc = 64'sd0;
            for (j=0;j<STATES;j=j+1)
                acc = acc + projected[i*STATES+j];
            skip_product = $signed(skip[i*WIDTH +: WIDTH]) * $signed(u[i*WIDTH +: WIDTH]);
            skip_ext = {{(64-2*WIDTH){skip_product[2*WIDTH-1]}},skip_product};
            acc = acc + round_shift(skip_ext,FRAC);
            next_y[i*WIDTH +: WIDTH] = clip_word(acc);
        end
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            y <= {CHANNELS*WIDTH{1'b0}};
        end else if (in_ready) begin
            out_valid <= in_valid;
            if (in_valid) y <= next_y;
        end
    end
endmodule
