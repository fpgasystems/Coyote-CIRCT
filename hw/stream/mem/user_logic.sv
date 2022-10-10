`timescale 1ns / 1ps

import lynxTypes::*;

`include "axi_macros.svh"
`include "lynx_macros.svh"

/**
 * User logic
 * 
 */
module design_user_logic_c0_{{c}} (
    // AXI4L CONTROL
    AXI4L.s                     axi_ctrl,

    // DESCRIPTOR BYPASS
    metaIntf.m			        bpss_rd_req,
    metaIntf.m			        bpss_wr_req,
    metaIntf.s                  bpss_rd_done,
    metaIntf.s                  bpss_wr_done,

    // AXI4S HOST STREAMS
    AXI4SR.s                    axis_host_sink,
    AXI4SR.m                    axis_host_src,

    // Clock and reset
    input  wire                 aclk,
    input  wire[0:0]            aresetn
);

/* -- Tie-off unused interfaces and signals ----------------------------- */
//always_comb axi_ctrl.tie_off_s();
//always_comb axis_host_sink.tie_off_s();
//always_comb axis_host_src.tie_off_m();
always_comb bpss_rd_req.tie_off_m();
//always_comb bpss_wr_req.tie_off_m();
//always_comb bpss_rd_done.tie_off_s();
//always_comb bpss_wr_done.tie_off_s();


/* -- USER LOGIC -------------------------------------------------------- */
assign bpss_wr_done.ready = 1'b1;
assign bpss_rd_done.ready = 1'b1;


// --------------------------------------------------------------------------------------
// Circuit
// --------------------------------------------------------------------------------------
AXI4SR axis_sink ();
AXI4SR axis_src ();
AXI4SR axis_s1 ();
AXI4SR axis_s2 ();

metaIntf #(.STYPE(dist_t)) s_dist ();
metaIntf #(.STYPE(dist_t)) m_dist ();

`AXISR_ASSIGN(axis_host_sink, axis_sink)
`AXISR_ASSIGN(axis_src, axis_host_src)

top_config_{{c}}_0 inst_top_{{c}} (
    .clock(aclk),
    .reset(~aresetn),
    
    // Sink
    .in0_valid(axis_sink.tvalid),
    .in0_ready(axis_sink.tready),
    .in0_data_field0(axis_sink.tdata),
    .in0_data_field1(axis_sink.tlast),

    .in1_valid(m_dist.valid),
    .in1_ready(m_dist.ready),
    .in1_data_field0(m_dist.data.hit),
    .in1_data_field1(m_dist.data.last),
    
    // Src
    .out0_valid(s_dist.valid),
    .out0_ready(s_dist.ready),
    .out0_data_field0(s_dist.data.key),
    .out0_data_field1(s_dist.data.last),

    .out1_valid(axis_s1.tvalid),
    .out1_ready(axis_s1.tready),
    .out1_data_field0(axis_s1.tdata),
    .out1_data_field1(axis_s1.tlast)
);

// Queue out
axis_data_fifo_sender_agg_512 inst_data_fifo_sender_agg_{{c}} (
    .s_axis_aclk(aclk),
    .s_axis_aresetn(aresetn),
    .s_axis_tvalid(axis_s2.tvalid),
    .s_axis_tready(axis_s2.tready),
    .s_axis_tdata (axis_s2.tdata),
    .s_axis_tlast (axis_s2.tlast),
    .m_axis_tvalid(axis_src.tvalid),
    .m_axis_tready(axis_src.tready),
    .m_axis_tdata (axis_src.tdata),
    .m_axis_tlast (axis_src.tlast)
);
assign axis_src.tkeep = ~0;

// --------------------------------------------------------------------------------------
// Distinct hash
// --------------------------------------------------------------------------------------
distinct_top inst_distinct (
    .aclk(aclk),
    .aresetn(aresetn),
    .s_meta(s_dist),
    .m_meta(m_dist)
);

// --------------------------------------------------------------------------------------
// Slave
// --------------------------------------------------------------------------------------
logic [VADDR_BITS-1:0] vaddr;
logic [PID_BITS-1:0] pid;

dist_slave inst_dist_slave_{{c}} (
    .aclk(aclk),
    .aresetn(aresetn),
    .axi_ctrl(axi_ctrl),
    .vaddr(vaddr),
    .pid(pid)
);

// --------------------------------------------------------------------------------------
// FSM
// --------------------------------------------------------------------------------------

// Params
localparam integer PMTU_BEATS = PMTU_BYTES / (AXI_DATA_BITS / 8);
localparam integer BEAT_LOG_BITS = $clog2(AXI_DATA_BITS / 8);

// FSM
typedef enum logic[1:0]  {ST_FILL, ST_SEND, ST_SEND_LAST} state_t;
logic [1:0] state_C = ST_FILL, state_N;

// Regs
logic [31:0] cnt_q_C, cnt_q_N;
logic [VADDR_BITS-1:0] addr_C = 0, addr_N;

// REG
always_ff @(posedge aclk) begin: PROC_REG
if (aresetn == 1'b0) begin
    state_C <= ST_FILL;
    cnt_q_C <= 0;
    addr_C <= 0;
end
else
    state_C <= state_N;
    cnt_q_C <= cnt_q_N;
    addr_C <= addr_N;
end

// NSL
always_comb begin: NSL
    state_N = state_C;

    case(state_C) 
        ST_FILL: 
            if(axis_s1.tvalid & axis_s1.tready) begin
                if(axis_s1.tlast) begin
                    state_N = ST_SEND_LAST;
                end
                else begin
                    if(cnt_q_C == PMTU_BEATS - 1) begin
                        state_N = ST_SEND;
                    end
                end
            end

        ST_SEND, ST_SEND_LAST: 
            if(bpss_wr_req.ready) begin
                state_N = ST_FILL;
            end

    endcase
end

// DP
always_comb begin: DP
    cnt_q_N = cnt_q_C;
    addr_N = addr_C;

    // Bypass
    bpss_wr_req.valid = 1'b0;
    bpss_wr_req.data = 0;
    bpss_wr_req.data.vaddr = vaddr + addr_C;
    bpss_wr_req.data.len =  cnt_q_C << BEAT_LOG_BITS;
    bpss_wr_req.data.ctl = 1'b0;
    bpss_wr_req.data.pid = pid;

    // Handshake
    axis_s1.tready = 1'b0;
    axis_s2.tvalid = 1'b0;
    axis_s2.tdata = axis_s1.tdata;
    axis_s2.tlast = axis_s1.tlast;

    case(state_C) 
        ST_FILL: begin
            axis_s2.tvalid = axis_s1.tvalid;
            axis_s1.tready = axis_s2.tready;

            if(axis_s1.tvalid & axis_s1.tready) begin
                cnt_q_N = cnt_q_C + 1;
            end
        end

        ST_SEND: begin
            bpss_wr_req.valid = 1'b1;

            if(bpss_wr_req.ready) begin
                cnt_q_N = 0;
                addr_N = addr_C + (1 << BEAT_LOG_BITS);             
            end
        end

        ST_SEND_LAST: begin
            bpss_wr_req.valid = 1'b1;
            bpss_wr_req.data.ctl = 1'b1;

            if(bpss_wr_req.ready) begin
                cnt_q_N = 0;
                addr_N = 0;
            end
        end

    endcase
end

//
// DEBUG
//
ila_agg_0 inst_ila_agg_{{c}} (
    .clk(aclk),
    
    .probe0(axis_sink.tvalid),
    .probe1(axis_sink.tready),
    .probe2(axis_sink.tlast),
    .probe3(axis_sink.tdata), // 512
    
    .probe4(axis_src.tvalid),
    .probe5(axis_src.tready),
    .probe6(axis_src.tlast),
    .probe7(axis_src.tdata), // 512

    .probe8(axis_s1.tvalid),
    .probe9(axis_s1.tready),
    .probe10(axis_s1.tlast),

    .probe11(axis_s2.tvalid),
    .probe12(axis_s2.tready),
    .probe13(axis_s2.tlast),

    .probe14(s_dist.valid), 
    .probe15(s_dist.ready),
    .probe16(s_dist.data), // 66  

    .probe17(m_dist.valid), 
    .probe18(m_dist.ready),
    .probe19(m_dist.data), // 66  

    .probe20(state_C), // 2
    .probe21(cnt_q_C), // 32
    .probe22(addr_C), // 48

    .probe23(bpss_wr_req.valid),
    .probe24(bpss_wr_req.ready),
    .probe25(bpss_wr_req.data.vaddr), // 48
    .probe26(bpss_wr_req.data.len), // 28
    .probe27(bpss_wr_req.data.ctl),
    .probe28(bpss_wr_req.data.pid) // 6
);

logic [31:0] cnt_sink;
logic [31:0] cnt_src;

always_ff @(posedge aclk) begin
    if(~aresetn) begin
        cnt_sink <= 0;
        cnt_src <= 0;
    end
    else begin
        cnt_sink <= axis_sink.tvalid & axis_sink.tready ? cnt_sink + 1 : cnt_sink;
        cnt_src <= axis_src.tvalid & axis_src.tready ? cnt_src + 1 : cnt_src;
    end
end 

vio_agg_0 inst_vio_agg_{{c}} (
    .clk(aclk),
    .probe_in0(cnt_sink), // 32
    .probe_in1(cnt_src) // 32
);


endmodule