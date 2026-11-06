`timescale 1ns / 1ps
`include "precision_config.svh"

// Eight-token signed Q6.10 classifier. Each token is processed by the trained
// projections and causal convolution, the independently verified selective scan,
// and the trained output projection. The last token triggers the classifier.
// Weight memories contain checkpoint-derived signed Q6.10 parameters.
module mamba_full_top (
    input  wire clk,
    input  wire rst_n,
    input  wire in_valid,
    output wire in_ready,
    input  wire [1023:0] pixels,
    output reg out_valid,
    input  wire out_ready,
    output reg signed [159:0] logits,
    output reg [3:0] prediction
);
    localparam integer Q = 10;
    typedef enum logic [2:0] {
        IDLE = 3'd0, FEED = 3'd1, CONSUME = 3'd2,
        HEAD = 3'd3, OUTPUT = 3'd4
    } phase_t;

    phase_t phase;
    reg [2:0] token_index;
    reg [1023:0] input_frame;
    assign in_ready = phase == IDLE;

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

    // For negative arguments the arithmetic right shift implements floor();
    // the positive residual determines nearest-even rounding symmetrically.
    function automatic signed [63:0] roundq;
        input signed [63:0] value;
        input integer bits;
        reg signed [63:0] floor_v, residue, halfway;
        begin
            floor_v = value >>> bits;
            residue = value - (floor_v <<< bits);
            halfway = 64'sd1 <<< (bits-1);
            if ((residue > halfway) ||
                ((residue == halfway) && floor_v[0]))
                roundq = floor_v + 64'sd1;
            else
                roundq = floor_v;
        end
    endfunction

    function automatic signed [15:0] sat16;
        input signed [63:0] value;
        begin
            if (value > 64'sd32767) sat16 = 16'sh7fff;
            else if (value < -64'sd32768) sat16 = 16'sh8000;
            else sat16 = value[15:0];
        end
    endfunction

    function automatic signed [15:0] qm;
        input signed [15:0] a;
        input signed [15:0] b;
        reg signed [31:0] prod;
        begin
            prod = a * b;
            qm = sat16(roundq({{32{prod[31]}}, prod}, Q));
        end
    endfunction

    // Sign-extend explicitly to the signed 64-bit accumulator width.
    function automatic signed [63:0] sx16;
        input signed [15:0] value;
        begin
            sx16 = {{48{value[15]}}, value};
        end
    endfunction

    reg signed [15:0] history_1 [0:7];
    reg signed [15:0] history_2 [0:7];
    reg signed [15:0] held_projection [0:7];
    reg signed [15:0] held_gate [0:7];
    reg signed [15:0] token_mem [0:63];

    reg signed [15:0] x [0:7];
    reg signed [15:0] projected [0:7];
    reg signed [15:0] gated [0:7];
    reg signed [15:0] u [0:7];
    reg signed [15:0] b [0:3];
    reg signed [15:0] c [0:3];
    reg signed [15:0] dt [0:7];
    reg [1:0] dt_index [0:7];
    reg [127:0] scan_u;
    reg [511:0] scan_alpha, scan_excitation;
    reg [63:0] scan_c;
    reg [127:0] scan_skip;
    wire scan_ready;
    wire scan_valid;
    wire signed [127:0] scan_output;
    reg signed [63:0] accum;
    reg signed [63:0] conv_acc;
    reg signed [15:0] conv_term;
    reg signed [15:0] gb;
    integer d, i, j, n;
    reg [6:0] rom_index;

    // Trained front end and parameter lookup for one image token.
    always @* begin
        accum = 64'sd0;
        conv_acc = 64'sd0;
        conv_term = 16'sd0;
        gb = 16'sd0;
        rom_index = 0;
        scan_u = 128'd0;
        scan_alpha = 512'd0;
        scan_excitation = 512'd0;
        scan_c = 64'd0;
        scan_skip = 128'd0;
        for (d=0;d<8;d=d+1) begin
            x[d] = $signed(input_frame[(token_index*8+d)*16 +: 16]);
        end
        for (j=0;j<8;j=j+1) begin
            accum = 64'sd0;
            for (i=0;i<8;i=i+1)
                accum = accum + $signed(ip_w[j*8+i]) * $signed(x[i]);
            projected[j] = sat16(roundq(accum,Q) + sx16(ip_b[j]));
            accum = 64'sd0;
            for (i=0;i<8;i=i+1)
                accum = accum + $signed(gate_w[j*8+i]) * $signed(x[i]);
            gated[j] = sat16(roundq(accum,Q) + sx16(gate_b[j]));
        end
        for (d=0;d<8;d=d+1) begin
            conv_acc = sx16(conv_b[d]);
            conv_term = qm(projected[d], conv_w[d]);
            conv_acc = conv_acc + sx16(conv_term);
            conv_term = qm(history_1[d], conv_w[8+d]);
            conv_acc = conv_acc + sx16(conv_term);
            conv_term = qm(history_2[d], conv_w[16+d]);
            conv_acc = conv_acc + sx16(conv_term);
            u[d] = sat16(conv_acc);
            if (u[d] < 16'sd0) u[d] = 16'sd0;
            scan_u[d*16 +:16] = u[d];
            scan_skip[d*16 +:16] = skip_w[d];
        end
        for (j=0;j<4;j=j+1) begin
            accum = 64'sd0;
            for (i=0;i<8;i=i+1)
                accum = accum + $signed(b_w[j*8+i])*$signed(u[i]);
            b[j] = sat16(roundq(accum,Q)+sx16(b_b[j]));
            accum = 64'sd0;
            for (i=0;i<8;i=i+1)
                accum = accum + $signed(c_w[j*8+i])*$signed(u[i]);
            c[j] = sat16(roundq(accum,Q)+sx16(c_b[j]));
            scan_c[j*16 +:16] = c[j];
        end
        for (j=0;j<8;j=j+1) begin
            accum = 64'sd0;
            for (i=0;i<8;i=i+1)
                accum = accum + $signed(dt_w[j*8+i])*$signed(u[i]);
            dt[j] = sat16(roundq(accum,Q)+sx16(dt_b[j]));
            if (dt[j] <= -16'sd512) dt_index[j] = 2'd0;
            else if (dt[j] <= 16'sd0) dt_index[j] = 2'd1;
            else if (dt[j] <= 16'sd512) dt_index[j] = 2'd2;
            else dt_index[j] = 2'd3;
            for (n=0;n<4;n=n+1) begin
                rom_index = {dt_index[j], j[2:0], n[1:0]};
                scan_alpha[(j*4+n)*16 +:16] = alpha_rom[rom_index];
                gb = qm(gain_rom[rom_index],b[n]);
                scan_excitation[(j*4+n)*16 +:16] = qm(gb,u[j]);
            end
        end
    end

    precision_scan #(
        .STATE_FRAC_MAP(`DIGITS_STATE_FRAC_MAP)
    ) u_scan (
        .clk(clk), .rst_n(rst_n),
        .in_valid(phase==FEED), .in_ready(scan_ready),
        .sequence_start(token_index==3'd0),
        .u(scan_u), .alpha(scan_alpha), .excitation(scan_excitation),
        .c(scan_c), .skip(scan_skip),
        .out_valid(scan_valid), .out_ready(phase==CONSUME),
        .y(scan_output)
    );

    reg signed [15:0] gate_value [0:7];
    reg signed [15:0] gated_y [0:7];
    reg signed [15:0] token_next [0:7];
    reg signed [15:0] head_scores [0:9];
    reg [6:0] head_index;
    reg signed [63:0] head_accum;
    reg signed [15:0] best_score_reg;
    reg [3:0] best_idx_reg;
    wire [3:0] head_class = head_index[6:3];
    wire [2:0] head_tile = head_index[2:0];
    reg signed [63:0] head_tile_sum;
    wire signed [63:0] head_total = head_accum + head_tile_sum;
    reg signed [15:0] final_head_value;
    reg signed [15:0] y_word;
    reg signed [63:0] post_acc;
    integer p, q;

    // Consumption occurs one cycle after the recurrent-state update.
    // The final head runs for 80 cycles, sharing eight 16x16 multipliers
    // rather than instantiating 640 parallel head multipliers.
    always @* begin
        post_acc = 64'sd0;
        y_word = 16'sd0;
        head_tile_sum = 64'sd0;
        final_head_value = 16'sd0;
        for (p=0;p<8;p=p+1) begin
            gate_value[p] = gate_div6(held_gate[p]);
            y_word = scan_output[p*16 +: 16];
            gated_y[p] = qm(y_word, gate_value[p]);
        end
        for (p=0;p<8;p=p+1) begin
            post_acc = 64'sd0;
            for (q=0;q<8;q=q+1)
                post_acc = post_acc + $signed(out_w[p*8+q]) * $signed(gated_y[q]);
            token_next[p] = sat16(sx16(sat16(roundq(post_acc,Q)+sx16(out_b[p])))
                                   + sx16(held_projection[p]));
        end
        for (p=0;p<8;p=p+1)
            head_tile_sum = head_tile_sum +
                $signed(head_w[head_class*64+head_tile*8+p]) *
                $signed(token_mem[head_tile*8+p]);
        final_head_value = sat16(roundq(head_total,Q)+sx16(head_b[head_class]));
    end

    function automatic signed [15:0] gate_div6;
        input signed [15:0] v;
        reg signed [63:0] positive;
        reg signed [63:0] base;
        reg signed [63:0] rem;
        begin
            positive = sx16(v) + 64'sd3072;
            // Verilog signed division truncates toward zero. The clipped
            // result is zero when the dividend is negative.
            if (positive <= 64'sd0) gate_div6=16'sd0;
            else if (positive >= 64'sd6144) gate_div6=16'sd1024;
            else begin
                base=positive/64'sd6;
                rem=positive%64'sd6;
                if ((rem>64'sd3) || ((rem==64'sd3) && base[0]))
                    base=base+64'sd1;
                gate_div6=base[15:0];
            end
        end
    endfunction

    integer z;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            phase <= IDLE;
            token_index <= 3'd0;
            head_index <= 7'd0;
            head_accum <= 64'sd0;
            best_score_reg <= 16'sh8000;
            best_idx_reg <= 4'd0;
            input_frame <= 1024'd0;
            logits <= 160'd0;
            prediction <= 4'd0;
            out_valid <= 1'b0;
            for (z=0;z<8;z=z+1) begin
                history_1[z]<=16'sd0;
                history_2[z]<=16'sd0;
                held_projection[z]<=16'sd0;
                held_gate[z]<=16'sd0;
            end
            for (z=0;z<64;z=z+1) token_mem[z]<=16'sd0;
            for (z=0;z<10;z=z+1) head_scores[z]<=16'sd0;
        end else begin
            case (phase)
                IDLE: if (in_valid) begin
                    input_frame <= pixels;
                    token_index <= 3'd0;
                    out_valid <= 1'b0;
                    for (z=0;z<8;z=z+1) begin
                        history_1[z]<=16'sd0;
                        history_2[z]<=16'sd0;
                    end
                    phase <= FEED;
                end
                FEED: if (scan_ready) begin
                    for (z=0;z<8;z=z+1) begin
                        history_2[z] <= history_1[z];
                        history_1[z] <= projected[z];
                        held_projection[z] <= projected[z];
                        held_gate[z] <= gated[z];
                    end
                    phase <= CONSUME;
                end
                CONSUME: if (scan_valid) begin
                    for (z=0;z<8;z=z+1)
                        token_mem[token_index*8+z] <= token_next[z];
                    if (token_index==3'd7) begin
                        head_index <= 7'd0;
                        head_accum <= 64'sd0;
                        best_score_reg <= 16'sh8000;
                        best_idx_reg <= 4'd0;
                        phase <= HEAD;
                    end else begin
                        token_index <= token_index + 3'd1;
                        phase <= FEED;
                    end
                end
                HEAD: begin
                    if (head_index[2:0]==3'd7) begin
                        head_scores[head_class] <= final_head_value;
                        head_accum <= 64'sd0;
                        if (head_class==4'd0 || final_head_value > best_score_reg) begin
                            best_score_reg <= final_head_value;
                            best_idx_reg <= head_class;
                        end
                        if (head_class==4'd9) begin
                            for (z=0;z<9;z=z+1)
                                logits[z*16 +:16] <= head_scores[z];
                            logits[9*16 +:16] <= final_head_value;
                            prediction <= (final_head_value > best_score_reg)
                                        ? 4'd9 : best_idx_reg;
                            out_valid <= 1'b1;
                            phase <= OUTPUT;
                        end else head_index <= head_index + 7'd1;
                    end else begin
                        head_accum <= head_total;
                        head_index <= head_index + 7'd1;
                    end
                end
                OUTPUT: if (out_ready) begin
                    out_valid <= 1'b0;
                    phase <= IDLE;
                end
                default: phase <= IDLE;
            endcase
        end
    end
endmodule
