`timescale 1ns / 1ps
`include "precision_config.svh"

// Complete Q6.10 trained digit classifier, single 16x16 signed multiplier.
// Input accepts one 8x8 frame; output holds ten exact integer logits until
// out_ready. Per-token MACs are scheduled across cycles, not elaborated as
// parallel nested combinational loops. The verified precision_scan is retained.
module mamba_serial_top (
    input wire clk, rst_n,
    input wire in_valid,
    output wire in_ready,
    input wire [1023:0] pixels,
    output reg out_valid,
    input wire out_ready,
    output reg signed [159:0] logits,
    output reg [3:0] prediction
);
    localparam [3:0]
        IDLE=4'd0, PROJ=4'd1, CONV=4'd2, PARAM=4'd3,
        EXCITE_A=4'd4, EXCITE_B=4'd5, FEED=4'd6, CONSUME=4'd7,
        GATE=4'd8, POST=4'd9, HEAD=4'd10, OUTPUT=4'd11;
    reg [3:0] phase;
    assign in_ready = phase == IDLE;
    reg [1023:0] input_frame;
    reg [2:0] token_index;
    reg signed [15:0] ip_w [0:63];
    reg signed [15:0] ip_b [0:7];
    reg signed [15:0] gate_w [0:63];
    reg signed [15:0] gate_b [0:7];
    reg signed [15:0] conv_w [0:23];
    reg signed [15:0] conv_b [0:7];
    reg signed [15:0] b_w [0:31];
    reg signed [15:0] b_b [0:3];
    reg signed [15:0] c_w [0:31];
    reg signed [15:0] c_b [0:3];
    reg signed [15:0] dt_w [0:63];
    reg signed [15:0] dt_b [0:7];
    reg signed [15:0] out_w [0:63];
    reg signed [15:0] out_b [0:7];
    reg signed [15:0] head_w [0:639];
    reg signed [15:0] head_b [0:9];
    reg signed [15:0] skip_w [0:7];
    reg signed [15:0] alpha_rom [0:127];
    reg signed [15:0] gain_rom [0:127];

    initial begin
        $readmemh("weights/full_rom/in_proj_w.mem", ip_w);
        $readmemh("weights/full_rom/in_proj_b.mem", ip_b);
        $readmemh("weights/full_rom/gate_proj_w.mem", gate_w);
        $readmemh("weights/full_rom/gate_proj_b.mem", gate_b);
        $readmemh("weights/full_rom/conv_w.mem", conv_w);
        $readmemh("weights/full_rom/conv_b.mem", conv_b);
        $readmemh("weights/full_rom/b_proj_w.mem", b_w);
        $readmemh("weights/full_rom/b_proj_b.mem", b_b);
        $readmemh("weights/full_rom/c_proj_w.mem", c_w);
        $readmemh("weights/full_rom/c_proj_b.mem", c_b);
        $readmemh("weights/full_rom/dt_proj_w.mem", dt_w);
        $readmemh("weights/full_rom/dt_proj_b.mem", dt_b);
        $readmemh("weights/full_rom/out_proj_w.mem", out_w);
        $readmemh("weights/full_rom/out_proj_b.mem", out_b);
        $readmemh("weights/full_rom/head_w.mem", head_w);
        $readmemh("weights/full_rom/head_b.mem", head_b);
        $readmemh("weights/full_rom/skip.mem", skip_w);
        $readmemh("weights/full_rom/alpha.mem", alpha_rom);
        $readmemh("weights/full_rom/gain.mem", gain_rom);
    end


    // Exact Python oracle: signed nearest-even Q6.10 + saturating int16.
    function automatic signed [63:0] roundq10(input signed [63:0] v);
        reg signed [63:0] floor_v, rem_v;
        begin
            floor_v = v >>> 10;
            rem_v = v - (floor_v <<< 10);
            if ((rem_v > 64'sd512) ||
                ((rem_v == 64'sd512) && floor_v[0]))
                roundq10 = floor_v + 64'sd1;
            else roundq10 = floor_v;
        end
    endfunction
    function automatic signed [15:0] sat16(input signed [63:0] v);
        begin
            if (v > 64'sd32767) sat16=16'sh7fff;
            else if (v < -64'sd32768) sat16=16'sh8000;
            else sat16=v[15:0];
        end
    endfunction
    function automatic signed [63:0] sx16(input signed [15:0] v);
        sx16={{48{v[15]}},v};
    endfunction
    function automatic signed [15:0] qm(input signed [31:0] prod);
        qm=sat16(roundq10({{32{prod[31]}},prod}));
    endfunction
    function automatic signed [15:0] gate_div6(input signed [15:0] v);
        reg signed [17:0] positive;
        reg [12:0] quotient, remainder;
        begin
            positive = {{2{v[15]}},v} + 18'sd3072;
            if (positive <= 18'sd0) gate_div6=16'sd0;
            else if (positive >= 18'sd6144) gate_div6=16'sd1024;
            else begin
                quotient = positive[12:0] / 13'd6;
                remainder = positive[12:0] % 13'd6;
                if ((remainder>13'd3) ||
                    ((remainder==13'd3) && quotient[0]))
                    quotient=quotient+13'd1;
                gate_div6={3'b000,quotient};
            end
        end
    endfunction

    reg signed [15:0] history_1 [0:7], history_2 [0:7];
    reg signed [15:0] projected [0:7], gated [0:7], u [0:7];
    reg signed [15:0] b [0:3], c [0:3];
    reg [1:0] dt_index [0:7];
    reg signed [15:0] gated_y [0:7];
    reg signed [15:0] token_mem [0:63];
    reg [127:0] scan_u, scan_skip;
    reg [511:0] scan_alpha, scan_excitation;
    reg [63:0] scan_c;
    wire scan_ready, scan_valid;
    wire signed [127:0] scan_output;
    reg signed [127:0] scan_result;
    reg signed [15:0] gb_tmp;

    reg [2:0] out_index, in_index;
    reg [1:0] conv_tap, proj_kind, param_kind;
    reg [5:0] exc_index, head_feature;
    reg [3:0] head_class;
    reg signed [63:0] accum;
    reg signed [15:0] mul_a, mul_b;
    wire signed [31:0] mul_product=$signed(mul_a)*$signed(mul_b);
    wire signed [63:0] mul_extended={{32{mul_product[31]}},mul_product};
    wire signed [63:0] mac_next=accum+mul_extended;
    reg [6:0] rom_address;
    reg signed [15:0] best_score;
    reg [3:0] best_class;
    wire signed [15:0] conv_qm=qm(mul_product);
    wire signed [63:0] conv_sum=accum+sx16(conv_qm);
    wire signed [15:0] conv_clip=sat16(conv_sum);
    wire signed [15:0] post_proj=sat16(roundq10(mac_next)+sx16(out_b[out_index]));
    wire signed [15:0] post_residual=sat16(sx16(post_proj)+sx16(projected[out_index]));
    wire signed [15:0] head_final=sat16(roundq10(mac_next)+sx16(head_b[head_class]));

    // Only one multiplier is inferred in this datapath: every phase supplies
    // one operand pair to mul_a/mul_b. Never iterate over products in always_comb.
    always @* begin
        mul_a=16'sd0;
        mul_b=16'sd0;
        rom_address=7'd0;
        case (phase)
            PROJ: begin
                mul_a=$signed(input_frame[{token_index,in_index,4'b0000} +:16]);
                if (proj_kind==2'd0) mul_b=ip_w[{out_index,in_index}];
                else mul_b=gate_w[{out_index,in_index}];
            end
            CONV: begin
                case (conv_tap)
                    2'd0: mul_a=projected[out_index];
                    2'd1: mul_a=history_1[out_index];
                    default: mul_a=history_2[out_index];
                endcase
                mul_b=conv_w[{conv_tap,out_index}];
            end
            PARAM: begin
                mul_a=u[in_index];
                case (param_kind)
                    2'd0: mul_b=b_w[{out_index[1:0],in_index}];
                    2'd1: mul_b=c_w[{out_index[1:0],in_index}];
                    default: mul_b=dt_w[{out_index,in_index}];
                endcase
            end
            EXCITE_A: begin
                rom_address={dt_index[exc_index[4:2]],exc_index[4:0]};
                mul_a=gain_rom[rom_address];
                mul_b=b[exc_index[1:0]];
            end
            EXCITE_B: begin
                mul_a=gb_tmp;
                mul_b=u[exc_index[4:2]];
            end
            GATE: begin
                mul_a=scan_result[out_index*16 +:16];
                mul_b=gate_div6(gated[out_index]);
            end
            POST: begin
                mul_a=gated_y[in_index];
                mul_b=out_w[{out_index,in_index}];
            end
            HEAD: begin
                mul_a=token_mem[head_feature];
                mul_b=head_w[{head_class,head_feature}];
            end
            default: begin end
        endcase
    end

    precision_scan #(.STATE_FRAC_MAP(`DIGITS_STATE_FRAC_MAP)) u_scan (
        .clk(clk),.rst_n(rst_n),
        .in_valid(phase==FEED),.in_ready(scan_ready),
        .sequence_start(token_index==3'd0),
        .u(scan_u),.alpha(scan_alpha),.excitation(scan_excitation),
        .c(scan_c),.skip(scan_skip),
        .out_valid(scan_valid),.out_ready(phase==CONSUME),
        .y(scan_output)
    );

    integer k;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            phase<=IDLE; token_index<=3'd0;
            input_frame<=1024'd0; out_valid<=1'b0;
            logits<=160'd0;prediction<=4'd0;
            out_index<=3'd0;in_index<=3'd0;
            proj_kind<=2'd0;param_kind<=2'd0;conv_tap<=2'd0;
            exc_index<=6'd0;head_feature<=6'd0;head_class<=4'd0;
            accum<=64'sd0;gb_tmp<=16'sd0;
            scan_u<=128'd0;scan_skip<=128'd0;
            scan_alpha<=512'd0;scan_excitation<=512'd0;scan_c<=64'd0;
            scan_result<=128'd0;
            best_score<=16'sh8000;best_class<=4'd0;
            for (k=0;k<8;k=k+1) begin
                history_1[k]<=16'sd0;history_2[k]<=16'sd0;
                projected[k]<=16'sd0;gated[k]<=16'sd0;
                u[k]<=16'sd0;
                dt_index[k]<=2'd0;gated_y[k]<=16'sd0;
            end
            for (k=0;k<4;k=k+1) begin b[k]<=16'sd0;c[k]<=16'sd0;end
            for (k=0;k<64;k=k+1) token_mem[k]<=16'sd0;
        end else begin
            case (phase)
                IDLE: if (in_valid) begin
                    input_frame<=pixels;token_index<=3'd0;
                    out_valid<=1'b0;
                    proj_kind<=2'd0;out_index<=3'd0;in_index<=3'd0;
                    accum<=64'sd0;
                    for (k=0;k<8;k=k+1) begin
                        history_1[k]<=16'sd0;history_2[k]<=16'sd0;
                    end
                    phase<=PROJ;
                end
                PROJ: begin
                    if (in_index==3'd7) begin
                        if (proj_kind==2'd0)
                            projected[out_index]<=sat16(roundq10(mac_next)+sx16(ip_b[out_index]));
                        else gated[out_index]<=sat16(roundq10(mac_next)+sx16(gate_b[out_index]));
                        accum<=64'sd0;in_index<=3'd0;
                        if (out_index==3'd7) begin
                            out_index<=3'd0;
                            if (proj_kind==2'd0) proj_kind<=2'd1;
                            else begin
                                conv_tap<=2'd0;
                                accum<=sx16(conv_b[0]);
                                phase<=CONV;
                            end
                        end else out_index<=out_index+3'd1;
                    end else begin
                        accum<=mac_next;in_index<=in_index+3'd1;
                    end
                end
                CONV: begin
                    if (conv_tap==2'd2) begin
                        u[out_index] <= conv_clip[15] ? 16'sd0 : conv_clip;
                        accum<=64'sd0;conv_tap<=2'd0;
                        if (out_index==3'd7) begin
                            out_index<=3'd0;in_index<=3'd0;
                            param_kind<=2'd0;phase<=PARAM;
                        end else begin
                            out_index<=out_index+3'd1;
                            accum<=sx16(conv_b[out_index+3'd1]);
                        end
                    end else begin
                        accum<=conv_sum;conv_tap<=conv_tap+2'd1;
                    end
                end
                PARAM: begin
                    if (in_index==3'd7) begin
                        case (param_kind)
                            2'd0: b[out_index[1:0]]<=sat16(roundq10(mac_next)+sx16(b_b[out_index[1:0]]));
                            2'd1: c[out_index[1:0]]<=sat16(roundq10(mac_next)+sx16(c_b[out_index[1:0]]));
                            default: begin
                                if (sat16(roundq10(mac_next)+sx16(dt_b[out_index]))<=-16'sd512)
                                    dt_index[out_index]<=2'd0;
                                else if (sat16(roundq10(mac_next)+sx16(dt_b[out_index]))<=16'sd0)
                                    dt_index[out_index]<=2'd1;
                                else if (sat16(roundq10(mac_next)+sx16(dt_b[out_index]))<=16'sd512)
                                    dt_index[out_index]<=2'd2;
                                else dt_index[out_index]<=2'd3;
                            end
                        endcase
                        accum<=64'sd0;in_index<=3'd0;
                        if ((param_kind==2'd2 && out_index==3'd7) ||
                            (param_kind!=2'd2 && out_index==3'd3)) begin
                            out_index<=3'd0;
                            if (param_kind==2'd2) begin
                                exc_index<=6'd0;phase<=EXCITE_A;
                                for (k=0;k<8;k=k+1) begin
                                    scan_u[k*16 +:16]<=u[k];
                                    scan_skip[k*16 +:16]<=skip_w[k];
                                end
                                for (k=0;k<4;k=k+1) scan_c[k*16 +:16]<=c[k];
                            end else param_kind<=param_kind+2'd1;
                        end else out_index<=out_index+3'd1;
                    end else begin
                        accum<=mac_next;in_index<=in_index+3'd1;
                    end
                end
                EXCITE_A: begin
                    gb_tmp<=qm(mul_product);
                    // rom_address is a 7-bit registered-index-dependent mux.
                    scan_alpha[exc_index*16 +:16]<=alpha_rom[rom_address];
                    phase<=EXCITE_B;
                end
                EXCITE_B: begin
                    scan_excitation[exc_index*16 +:16]<=qm(mul_product);
                    if (exc_index==6'd31) begin phase<=FEED;end
                    else begin exc_index<=exc_index+6'd1;phase<=EXCITE_A;end
                end
                FEED: if (scan_ready) begin
                    // History advances only with the accepted scan token.
                    // The current convolution has already consumed old history.
                    for (k=0;k<8;k=k+1) begin
                        history_2[k]<=history_1[k];
                        history_1[k]<=projected[k];
                    end
                    phase<=CONSUME;
                end
                CONSUME: if (scan_valid) begin
                    scan_result<=scan_output;
                    out_index<=3'd0;phase<=GATE;
                end
                GATE: begin
                    gated_y[out_index]<=qm(mul_product);
                    if (out_index==3'd7) begin
                        out_index<=3'd0;in_index<=3'd0;accum<=64'sd0;
                        phase<=POST;
                    end else out_index<=out_index+3'd1;
                end
                POST: begin
                    if (in_index==3'd7) begin
                        token_mem[{token_index,out_index}]<=post_residual;
                        accum<=64'sd0;in_index<=3'd0;
                        if (out_index==3'd7) begin
                            out_index<=3'd0;
                            if (token_index==3'd7) begin
                                head_class<=4'd0;head_feature<=6'd0;
                                best_score<=16'sh8000;best_class<=4'd0;
                                phase<=HEAD;
                            end else begin
                                token_index<=token_index+3'd1;
                                proj_kind<=2'd0;phase<=PROJ;
                            end
                        end else out_index<=out_index+3'd1;
                    end else begin accum<=mac_next;in_index<=in_index+3'd1;end
                end
                HEAD: begin
                    if (head_feature==6'd63) begin
                        logits[{head_class,4'b0000} +:16]<=head_final;
                        accum<=64'sd0;head_feature<=6'd0;
                        if (head_class==4'd0 || head_final>best_score) begin
                            best_score<=head_final;best_class<=head_class;
                        end
                        if (head_class==4'd9) begin
                            prediction<=(head_final>best_score) ? 4'd9 : best_class;
                            out_valid<=1'b1;phase<=OUTPUT;
                        end else head_class<=head_class+4'd1;
                    end else begin
                        accum<=mac_next;head_feature<=head_feature+6'd1;
                    end
                end
                OUTPUT: if (out_ready) begin
                    out_valid<=1'b0;phase<=IDLE;
                end
                default: phase<=IDLE;
            endcase
        end
    end
endmodule
