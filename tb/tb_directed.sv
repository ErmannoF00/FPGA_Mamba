`timescale 1ns / 1ps

// Directed, independently calculated arithmetic and handshake cases.
module tb_directed;
    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg rst_n = 1'b1;
    reg in_valid = 1'b0;
    wire in_ready;
    reg sequence_start = 1'b0;
    reg signed [15:0] x = 16'sd0;
    reg signed [15:0] a = 16'sd0;
    reg signed [15:0] b = 16'sd0;
    reg signed [15:0] c = 16'sd0;
    wire out_valid;
    reg out_ready = 1'b0;
    wire signed [15:0] y;

    selective_scan #(.LANES(1)) dut (
        .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .in_ready(in_ready),
        .sequence_start(sequence_start), .x(x), .a(a), .b(b), .c(c),
        .out_valid(out_valid), .out_ready(out_ready), .y(y)
    );

    task automatic transfer;
        input signed [15:0] x_i;
        input signed [15:0] a_i;
        input signed [15:0] b_i;
        input signed [15:0] c_i;
        input seq_i;
        input signed [15:0] expected;
        begin
            @(negedge clk);
            x = x_i; a = a_i; b = b_i; c = c_i;
            sequence_start = seq_i; in_valid = 1'b1; out_ready = 1'b1;
            #1;
            if (!in_ready) $fatal(1, "input unexpectedly stalled");
            @(posedge clk); #1;
            if (out_valid !== 1'b1 || y !== expected)
                $fatal(1, "directed transfer: output=%0d expected=%0d", y, expected);
        end
    endtask

    initial begin
        #1 rst_n = 1'b0;
        #1;
        if (out_valid !== 1'b0 || y !== 16'sd0 || !in_ready)
            $fatal(1, "asynchronous reset failed");
        @(negedge clk);
        rst_n = 1'b1;

        // h=1.0; y=1.0+0.5=1.5.
        transfer(16'sd4096, 16'sd0, 16'sd4096,
                 16'sd4096, 1'b1, 16'sd6144);

        // h=0.5*1+0.5*1=1; output remains 1.5.
        transfer(16'sd4096, 16'sd2048, 16'sd2048,
                 16'sd4096, 1'b0, 16'sd6144);

        // Block output for several cycles; input pins must be ignored.
        @(negedge clk);
        out_ready = 1'b0;
        sequence_start = 1'b1;
        x = -16'sd4096; a = 16'sd4096;
        b = 16'sd4096; c = 16'sd4096;
        repeat (12) begin
            @(posedge clk); #1;
            if (in_ready !== 1'b0 || out_valid !== 1'b1 || y !== 16'sd6144)
                $fatal(1, "state/output changed during backpressure");
        end

        // Clear old state through a=0, produce -1.5.
        transfer(-16'sd4096, 16'sd0, 16'sd4096,
                 16'sd4096, 1'b0, -16'sd6144);

        // Positive and negative saturation (no intermediate wrap).
        transfer(16'sd32767, 16'sd0, 16'sd32767,
                 16'sd32767, 1'b0, 16'sd32767);
        transfer(-16'sd32768, 16'sd0, 16'sd32767,
                 16'sd32767, 1'b0, -16'sd32768);

        // sequence_start clears the prior saturated state before this token.
        transfer(16'sd4096, 16'sd4096, 16'sd0,
                 16'sd4096, 1'b1, 16'sd2048);

        @(negedge clk);
        rst_n = 1'b0;
        #1;
        if (out_valid !== 1'b0 || y !== 16'sd0 || !in_ready)
            $fatal(1, "asynchronous reset did not flush output");
        $display("PASS: directed arithmetic, saturation, reset and backpressure");
        $finish;
    end
endmodule
