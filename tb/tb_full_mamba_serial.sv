`timescale 1ns / 1ps

// End-to-end scoreboard: only 8x8 pixels are applied to the RTL.
// No intermediate activation, projection or recurrent coefficient is supplied
// by the testbench. Expected logits are calculated independently in Python.
module tb_full_mamba_serial;
    reg clk=1'b0;
    always #5 clk=~clk;
    reg rst_n=1'b1;
    reg in_valid=1'b0;
    wire in_ready;
    reg [1023:0] pixels=1024'd0;
    wire out_valid;
    reg out_ready=1'b0;
    wire signed [159:0] logits;
    wire [3:0] prediction;
    reg [1023:0] expected_pixels;
    reg [159:0] expected_logits;
    integer expected_label;
    integer expected_prediction;
    integer file_id;
    integer parsed;
    integer cycles;
    integer sample;
    integer max_samples;
    reg [1023:0] vector_path;

    mamba_serial_top dut (
        .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .in_ready(in_ready),
        .pixels(pixels), .out_valid(out_valid), .out_ready(out_ready),
        .logits(logits), .prediction(prediction)
    );
    initial begin
        if (!$value$plusargs("VECTORS=%s",vector_path))
            vector_path="weights/full_rom/images_expected.txt";
        if (!$value$plusargs("SAMPLES=%d",max_samples))
            max_samples=50;
        file_id=$fopen(vector_path,"r");
        if (file_id==0) $fatal(1,"Cannot open full-inference vectors");
        #1 rst_n=1'b0;
        #11 rst_n=1'b1;
        sample=0;
        while (sample < max_samples) begin
            parsed=$fscanf(file_id,"%h %h %d %d\n",expected_pixels,
                           expected_logits,expected_label,expected_prediction);
            if (parsed != 4)
                $fatal(1,"Missing/malformed input vector sample=%0d parsed=%0d",sample,parsed);
            @(negedge clk);
            if (!in_ready) $fatal(1,"Unexpected not-ready before sample %0d",sample);
            pixels=expected_pixels;
            in_valid=1'b1;
            @(negedge clk);
            in_valid=1'b0;
            cycles=0;
            while (!out_valid && cycles<12000) begin
                @(negedge clk);
                cycles=cycles+1;
            end
            if (!out_valid) $fatal(1,"Timeout sample=%0d",sample);
            if (logits !== expected_logits)
                $fatal(1,"Logits mismatch sample=%0d actual=%h expected=%h",
                       sample,logits,expected_logits);
            if (prediction !== expected_prediction[3:0])
                $fatal(1,"Prediction mismatch sample=%0d actual=%0d expected=%0d",
                       sample,prediction,expected_prediction);
            // Hold output under backpressure, checking stable data.
            repeat (2) begin
                @(negedge clk);
                if (!out_valid || logits !== expected_logits ||
                    prediction !== expected_prediction[3:0])
                    $fatal(1,"Output stalled incorrectly sample=%0d",sample);
            end
            out_ready=1'b1;
            @(negedge clk);
            out_ready=1'b0;
            if (out_valid) $fatal(1,"Output valid not cleared sample=%0d",sample);
            sample=sample+1;
        end
        $fclose(file_id);
        $display("PASS SERIAL-MAC FULL RTL: %0d complete digit predictions, exact integer logits",sample);
        $finish;
    end
endmodule
