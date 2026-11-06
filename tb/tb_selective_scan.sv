`timescale 1ns / 1ps

// Randomized cycle-by-cycle transaction scoreboard.
// File parsing uses temporary variables. Direct $fscanf writes to DUT-facing
// regs are avoided because some simulators do not schedule combinational
// reevaluation consistently for such assignments.
module tb_selective_scan;
    parameter integer LANES = 4;
    localparam integer WIDTH = 16;

    reg clk = 1'b0;
    reg rst_n = 1'b1;
    reg in_valid = 1'b0;
    wire in_ready;
    reg sequence_start = 1'b0;
    reg signed [LANES*WIDTH-1:0] x = 0;
    reg signed [LANES*WIDTH-1:0] a = 0;
    reg signed [LANES*WIDTH-1:0] b = 0;
    reg signed [LANES*WIDTH-1:0] c = 0;
    wire out_valid;
    reg out_ready = 1'b0;
    wire signed [LANES*WIDTH-1:0] y;

    integer raw_reset, raw_valid, raw_ready, raw_start;
    reg signed [LANES*WIDTH-1:0] raw_x, raw_a, raw_b, raw_c;
    integer expected_ready, expected_valid;
    reg signed [LANES*WIDTH-1:0] expected_y;
    integer fd, parsed, cycles;
    string vector_file;

    selective_scan #(.LANES(LANES)) dut (
        .clk(clk), .rst_n(rst_n),
        .in_valid(in_valid), .in_ready(in_ready),
        .sequence_start(sequence_start),
        .x(x), .a(a), .b(b), .c(c),
        .out_valid(out_valid), .out_ready(out_ready), .y(y)
    );

    always #5 clk = ~clk;

    initial begin
        // An explicit falling edge initializes asynchronous reset registers.
        #1 rst_n = 1'b0;
        if (!$value$plusargs("VECTORS=%s", vector_file))
            vector_file = "build/scan_vectors.txt";
        fd = $fopen(vector_file, "r");
        if (fd == 0)
            $fatal(1, "Cannot open vectors: %s", vector_file);

        cycles = 0;
        while (!$feof(fd)) begin
            @(negedge clk);
            parsed = $fscanf(fd, "%d %d %d %d %h %h %h %h %d %d %h\n",
                raw_reset, raw_valid, raw_ready, raw_start,
                raw_x, raw_a, raw_b, raw_c,
                expected_ready, expected_valid, expected_y);
            if (parsed == 11) begin
                rst_n = raw_reset != 0;
                in_valid = raw_valid != 0;
                out_ready = raw_ready != 0;
                sequence_start = raw_start != 0;
                x = raw_x;
                a = raw_a;
                b = raw_b;
                c = raw_c;
                #1;
                if (in_ready !== (expected_ready != 0) ||
                    out_valid !== (expected_valid != 0) ||
                    y !== expected_y)
                    $fatal(1,
                        "cycle=%0d: ready=%b/%0d valid=%b/%0d y=%h/%h x=%h a=%h b=%h c=%h",
                        cycles, in_ready, expected_ready,
                        out_valid, expected_valid, y, expected_y,
                        x, a, b, c);
                cycles = cycles + 1;
            end else if (!$feof(fd)) begin
                $fatal(1, "Malformed row at cycle %0d (read %0d fields)",
                       cycles, parsed);
            end
        end
        if (cycles < 300)
            $fatal(1, "Only %0d cycles checked", cycles);
        $fclose(fd);
        @(negedge clk);
        rst_n = 1'b1;
        in_valid = 1'b0;
        out_ready = 1'b1;
        repeat (2) @(posedge clk);
        @(negedge clk);
        if (out_valid || !in_ready)
            $fatal(1, "Output failed to drain");
        $display("PASS: %0d cycle comparisons, LANES=%0d", cycles, LANES);
        $finish;
    end
endmodule
