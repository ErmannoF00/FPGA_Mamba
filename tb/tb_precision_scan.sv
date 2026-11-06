`timescale 1ns / 1ps
`include "precision_config.svh"

// Clocked trace comparison with transactions extracted from trained-model
// handwritten-digit examples. All expected outputs come from scan_reference.py.
module tb_precision_scan;
    localparam integer CHANNELS=8;
    localparam integer STATES=4;
    localparam integer WIDTH=16;
    reg clk=1'b0;
    reg rst_n=1'b1;
    reg in_valid=1'b0;
    reg out_ready=1'b0;
    reg sequence_start=1'b0;
    reg signed [CHANNELS*WIDTH-1:0] u=0;
    reg signed [CHANNELS*STATES*WIDTH-1:0] alpha=0;
    reg signed [CHANNELS*STATES*WIDTH-1:0] excitation=0;
    reg signed [STATES*WIDTH-1:0] c=0;
    reg signed [CHANNELS*WIDTH-1:0] skip=0;
    wire in_ready;
    wire out_valid;
    wire signed [CHANNELS*WIDTH-1:0] y;

    integer raw_reset,raw_valid,raw_ready,raw_start;
    reg signed [CHANNELS*WIDTH-1:0] raw_u,raw_skip;
    reg signed [CHANNELS*STATES*WIDTH-1:0] raw_alpha,raw_excitation;
    reg signed [STATES*WIDTH-1:0] raw_c;
    integer expected_ready,expected_valid;
    reg signed [CHANNELS*WIDTH-1:0] expected_y;
    integer fd,parsed,cycles;
    string vector_file;
    string trace_file;
    integer trace_fd,trace_count;

    precision_scan #(.STATE_FRAC_MAP(`DIGITS_STATE_FRAC_MAP)) dut (
        .clk(clk),.rst_n(rst_n),.in_valid(in_valid),.in_ready(in_ready),
        .sequence_start(sequence_start),.u(u),.alpha(alpha),
        .excitation(excitation),.c(c),.skip(skip),
        .out_valid(out_valid),.out_ready(out_ready),.y(y)
    );
    always #5 clk=~clk;
    initial begin
        #1 rst_n=1'b0;
        if (!$value$plusargs("VECTORS=%s",vector_file))
            vector_file="build/model_scan_vectors.txt";
        fd=$fopen(vector_file,"r");
        if (fd==0)$fatal(1,"Cannot open vector file %s",vector_file);
        cycles=0;
        trace_fd=0;
        trace_count=0;
        if ($value$plusargs("TRACE=%s",trace_file)) begin
            trace_fd=$fopen(trace_file,"w");
            if (trace_fd==0) $fatal(1,"Cannot open trace file %s",trace_file);
        end
        while (!$feof(fd)) begin
            @(negedge clk);
            parsed=$fscanf(fd,"%d %d %d %d %h %h %h %h %h %d %d %h\n",
                raw_reset,raw_valid,raw_ready,raw_start,
                raw_u,raw_alpha,raw_excitation,raw_c,raw_skip,
                expected_ready,expected_valid,expected_y);
            if (parsed==12) begin
                rst_n=raw_reset!=0;
                in_valid=raw_valid!=0;
                out_ready=raw_ready!=0;
                sequence_start=raw_start!=0;
                u=raw_u;alpha=raw_alpha;excitation=raw_excitation;
                c=raw_c;skip=raw_skip;
                #1;
                if (in_ready !== (expected_ready!=0) ||
                    out_valid !== (expected_valid!=0) || y !== expected_y)
                    $fatal(1,"cycle=%0d ready=%b/%0d valid=%b/%0d y=%h/%h",
                        cycles,in_ready,expected_ready,out_valid,expected_valid,
                        y,expected_y);
                // Record exactly the output transactions consumed by the host.
                // Sampling occurs before the rising edge that completes transfer.
                if (trace_fd!=0 && rst_n && out_valid && out_ready) begin
                    $fdisplay(trace_fd,"%h",y);
                    trace_count=trace_count+1;
                end
                cycles=cycles+1;
            end else if (!$feof(fd))
                $fatal(1,"Malformed vector row at %0d; read %0d fields",cycles,parsed);
        end
        if (cycles<300) $fatal(1,"Insufficient scan cycles (%0d)",cycles);
        $fclose(fd);
        if (trace_fd!=0) begin
            if (trace_count<1) $fatal(1,"No consumed outputs were recorded");
            $fclose(trace_fd);
            $display("TRACE: %0d consumed scan outputs written to %s",trace_count,trace_file);
        end
        @(negedge clk);
        in_valid=1'b0;out_ready=1'b1;rst_n=1'b1;
        repeat (2) @(posedge clk);
        @(negedge clk);
        if (out_valid || !in_ready) $fatal(1,"Output failed to drain");
        $display("PASS: %0d trained-model scan cycles",cycles);
        $finish;
    end
endmodule
