// ======================
// Description: Testbench for mamba_top.sv
// Includes stimulus, waveform dump, and output tracking
// ======================

`timescale 1ns / 1ps

module tb_mamba_wrapper;

    parameter WIDTH = 16;
    parameter CLK_PERIOD = 10;

    logic clk;
    logic rst_n;
    logic [WIDTH-1:0] input_data;
    logic [WIDTH-1:0] output_data;

    mamba_top dut (
        .clk(clk),
        .rst_n(rst_n),
        .input_data(input_data),
        .output_data(output_data)
    );

    // Clock generator
    always #(CLK_PERIOD/2) clk = ~clk;

    // Input stimulus
    initial begin
        $dumpfile("waveform/mamba_wrapper_tb.vcd");
        $dumpvars(0, tb_mamba_wrapper);

        clk     = 0;
        rst_n   = 0;
        input_data = 0;

        #20 rst_n = 1;

        // Send input values (Q1.15 encoded)
        repeat (16) begin
            @(posedge clk);
            input_data = $random % 32768; 
        end

        // Wait some cycles
        repeat (20) @(posedge clk);

        $finish;
    end

endmodule
