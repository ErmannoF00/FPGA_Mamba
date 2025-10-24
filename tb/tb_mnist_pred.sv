// ======================
// File: tb_mnist_pred.sv
// Description: Top-level testbench for MNIST classification
// ======================

`timescale 1ns/1ps

module tb_mnist_predict;

    parameter WIDTH = 16;
    parameter IN_DIM = 784;
    parameter OUT_DIM = 10;

    logic clk, rst_n, valid_in;
    logic [WIDTH-1:0] x [0:IN_DIM-1];
    logic [WIDTH-1:0] y [0:OUT_DIM-1];
    logic valid_out;
    logic [3:0] predicted_class;
    logic valid_class;

    // Clock gen
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    initial begin
        $dumpfile("waveform/mnist_predict.vcd");
        $dumpvars(0, tb_mnist_predict);
        $readmemb("data/input_image.mem", x);

        rst_n = 0; valid_in = 0;
        #20 rst_n = 1; valid_in = 1;
        #10 valid_in = 0;
        #1000 $finish;
    end

    // Mamba pipeline wrapper
    mamba_wrapper uut (
        .clk(clk), .rst_n(rst_n),
        .valid_in(valid_in), .x(x),
        .valid_out(valid_out), .y(y)
    );

    // Argmax
    argmax argmax_inst (
        .clk(clk),
        .valid_in(valid_out),
        .in_vec(y),
        .out_class(predicted_class),
        .valid_out(valid_class)
    );

    // Logger
    output_logger logger (
        .clk(clk),
        .valid(valid_out),
        .data(y)
    );

endmodule
