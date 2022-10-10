`timescale 1ns / 1ps

import lynxTypes::*;

`include "axi_macros.svh"
`include "lynx_macros.svh"

{{d}}`define DBG_PROBES_C0_{{c}}

/**
 * User logic
 * 
 */
module design_user_logic_c0_0 (
// AXI4L CONTROL
    AXI4L.s                     axi_ctrl,

`ifdef EN_BPSS
    // DESCRIPTOR BYPASS
    metaIntf.m			        bpss_rd_req,
    metaIntf.m			        bpss_wr_req,
    metaIntf.s                  bpss_rd_done,
    metaIntf.s                  bpss_wr_done,

`endif
`ifdef EN_STRM
    // AXI4S HOST STREAMS
    AXI4SR.s                    axis_host_sink,
    AXI4SR.m                    axis_host_src,
`endif
`ifdef EN_MEM
    // AXI4S CARD STREAMS
    AXI4SR.s                    axis_card_sink,
    AXI4SR.m                    axis_card_src,
`endif
`ifdef EN_RDMA_0
    // RDMA QSFP0 CMD
    metaIntf.s			        rdma_0_rd_req,
    metaIntf.s 			        rdma_0_wr_req,

    // AXI4S RDMA QSFP0 STREAMS
    AXI4SR.s                    axis_rdma_0_sink,
    AXI4SR.m                    axis_rdma_0_src,
`ifdef EN_RPC
    // RDMA QSFP1 SQ
    metaIntf.m 			        rdma_0_sq,
    metaIntf.s                  rdma_0_rq,
`endif
`endif
`ifdef EN_RDMA_1
    // RDMA QSFP1 CMD
    metaIntf.s			        rdma_1_rd_req,
    metaIntf.s 			        rdma_1_wr_req,

    // AXI4S RDMA QSFP1 STREAMS
    AXI4SR.s                    axis_rdma_1_sink,
    AXI4SR.m                    axis_rdma_1_src,
`ifdef EN_RPC
    // RDMA QSFP1 SQ
    metaIntf.m 			        rdma_1_sq,
    metaIntf.s                  rdma_1_rq,
`endif
`endif
`ifdef EN_TCP_0
    // TCP/IP QSFP0 CMD
    metaIntf.m			        tcp_0_listen_req,
    metaIntf.s			        tcp_0_listen_rsp,
    metaIntf.m			        tcp_0_open_req,
    metaIntf.s			        tcp_0_open_rsp,
    metaIntf.m			        tcp_0_close_req,
    metaIntf.s			        tcp_0_notify,
    metaIntf.m			        tcp_0_rd_pkg,
    metaIntf.s			        tcp_0_rx_meta,
    metaIntf.m			        tcp_0_tx_meta,
    metaIntf.s			        tcp_0_tx_stat,

    // AXI4S TCP/IP QSFP0 STREAMS
    AXI4SR.s                    axis_tcp_0_sink,
    AXI4SR.m                    axis_tcp_0_src,
`endif
`ifdef EN_TCP_1
    // TCP/IP QSFP1 CMD
    metaIntf.m			        tcp_1_listen_req,
    metaIntf.s			        tcp_1_listen_rsp,
    metaIntf.m			        tcp_1_open_req,
    metaIntf.s			        tcp_1_open_rsp,
    metaIntf.m			        tcp_1_close_req,
    metaIntf.s			        tcp_1_notify,
    metaIntf.m			        tcp_1_rd_pkg,
    metaIntf.s			        tcp_1_rx_meta,
    metaIntf.m			        tcp_1_tx_meta,
    metaIntf.s			        tcp_1_tx_stat,

    // AXI4S TCP/IP QSFP1 STREAMS
    AXI4SR.s                    axis_tcp_1_sink, 
    AXI4SR.m                    axis_tcp_1_src,
`endif
    // Clock and reset
    input  wire                 aclk,
    input  wire[0:0]            aresetn
);

/* -- Tie-off unused interfaces and signals ----------------------------- */
//always_comb axi_ctrl.tie_off_s();
`ifdef EN_BPSS
//always_comb bpss_rd_req.tie_off_m();
//always_comb bpss_wr_req.tie_off_m();
always_comb bpss_rd_done.tie_off_s();
always_comb bpss_wr_done.tie_off_s();
`endif
`ifdef EN_STRM
//always_comb axis_host_sink.tie_off_s();
//always_comb axis_host_src.tie_off_m();
`endif
`ifdef EN_MEM
always_comb axis_card_sink.tie_off_s();
always_comb axis_card_src.tie_off_m();
`endif
`ifdef EN_RDMA_0
always_comb rdma_0_rd_req.tie_off_s();
always_comb rdma_0_wr_req.tie_off_s();
always_comb axis_rdma_0_sink.tie_off_s();
always_comb axis_rdma_0_src.tie_off_m();
`ifdef EN_RPC
always_comb rdma_0_sq.tie_off_m();
always_comb rdma_0_rq.tie_off_s();
`endif
`endif
`ifdef EN_RDMA_1
always_comb rdma_1_rd_req.tie_off_s();
always_comb rdma_1_wr_req.tie_off_s();
always_comb axis_rdma_1_sink.tie_off_s();
always_comb axis_rdma_1_src.tie_off_m();
`ifdef EN_RPC
always_comb rdma_1_sq.tie_off_m();
always_comb rdma_1_rq.tie_off_s();
`endif
`endif
`ifdef EN_TCP_0
always_comb tcp_0_listen_req.tie_off_m();
always_comb tcp_0_listen_rsp.tie_off_s();
always_comb tcp_0_open_req.tie_off_m();
always_comb tcp_0_open_rsp.tie_off_s();
always_comb tcp_0_close_req.tie_off_m();
always_comb tcp_0_notify.tie_off_s();
always_comb tcp_0_rd_pkg.tie_off_m();
always_comb tcp_0_rx_meta.tie_off_s();
always_comb tcp_0_tx_meta.tie_off_m();
always_comb tcp_0_tx_stat.tie_off_s();
always_comb axis_tcp_0_sink.tie_off_s();
always_comb axis_tcp_0_src.tie_off_m();
`endif
`ifdef EN_TCP_1
always_comb tcp_1_listen_req.tie_off_m();
always_comb tcp_1_listen_rsp.tie_off_s();
always_comb tcp_1_open_req.tie_off_m();
always_comb tcp_1_open_rsp.tie_off_s();
always_comb tcp_1_close_req.tie_off_m();
always_comb tcp_1_notify.tie_off_s();
always_comb tcp_1_rd_pkg.tie_off_m();
always_comb tcp_1_rx_meta.tie_off_s();
always_comb tcp_1_tx_meta.tie_off_m();
always_comb tcp_1_tx_stat.tie_off_s();
always_comb axis_tcp_1_sink.tie_off_s();
always_comb axis_tcp_1_src.tie_off_m();
`endif

/* -- USER LOGIC -------------------------------------------------------- */
// --------------------------------------------------------------------------------------
// Tables
// --------------------------------------------------------------------------------------
AXI4SR axis_sink [2] ();
AXI4SR axis_src ();
AXI4SR axis_r0 ();
AXI4SR axis_r1 ();
AXI4SR axis_s0 ();

metaIntf #(.STYPE(ext_t)) s_join ();
metaIntf #(.STYPE(ext_t)) m_join ();
metaIntf #(.STYPE(ext_t)) s_hash ();
metaIntf #(.STYPE(ext_t)) m_hash ();

`AXISR_ASSIGN(axis_src, axis_host_src)

// --------------------------------------------------------------------------------------
// R
// --------------------------------------------------------------------------------------
top_r_0 inst_top_r (
    .clock(aclk),
    .reset(~aresetn),
    
    // Sink
    .in0_valid(axis_sink[0].tvalid),
    .in0_ready(axis_sink[0].tready),
    .in0_data_field0_field0(axis_sink[0].tdata[0*64+:64]),
    .in0_data_field0_field1(axis_sink[0].tdata[1*64+:64]),
    .in0_data_field0_field2(axis_sink[0].tdata[2*64+:64]),
    .in0_data_field0_field3(axis_sink[0].tdata[3*64+:64]),
    .in0_data_field0_field4(axis_sink[0].tdata[4*64+:64]),
    .in0_data_field0_field5(axis_sink[0].tdata[5*64+:64]),
    .in0_data_field0_field6(axis_sink[0].tdata[6*64+:64]),
    .in0_data_field0_field7(axis_sink[0].tdata[7*64+:64]),
    .in0_data_field1(axis_sink[0].tlast),

    .in1_valid(m_join.valid),
    .in1_ready(m_join.ready),
    .in1_data_field0(m_join.data.hit),
    .in1_data_field1(m_join.data.last),

    .in2_valid(m_hash.valid),
    .in2_ready(m_hash.ready),
    .in2_data_field0(m_hash.data.hit),
    .in2_data_field1(m_hash.data.last),
    
    // Src
    .out0_valid(axis_r0.tvalid),
    .out0_ready(axis_r0.tready),
    .out0_data_field0_field0(axis_r0.tdata[0*64+:64]),
    .out0_data_field0_field1(axis_r0.tdata[1*64+:64]),
    .out0_data_field1(axis_r0.tlast),

    .out1_valid(s_join.valid),
    .out1_ready(s_join.ready),
    .out1_data_field0(s_join.data.key),
    .out1_data_field1(s_join.data.last),

    .out2_valid(s_hash.valid),
    .out2_ready(s_hash.ready),
    .out2_data_field0(s_hash.data.key),
    .out2_data_field1(s_hash.data.last)
);

// --------------------------------------------------------------------------------------
// S
// --------------------------------------------------------------------------------------

top_s_0 inst_top_s (
    .clock(aclk),
    .reset(~aresetn),
    
    // Sink
    .in0_valid(axis_sink[1].tvalid),
    .in0_ready(axis_sink[1].tready),
    .in0_data_field0_field0(axis_sink[1].tdata[0*64+:64]),
    .in0_data_field0_field1(axis_sink[1].tdata[1*64+:64]),
    .in0_data_field0_field2(axis_sink[1].tdata[2*64+:64]),
    .in0_data_field0_field3(axis_sink[1].tdata[3*64+:64]),
    .in0_data_field0_field4(axis_sink[1].tdata[4*64+:64]),
    .in0_data_field0_field5(axis_sink[1].tdata[5*64+:64]),
    .in0_data_field0_field6(axis_sink[1].tdata[6*64+:64]),
    .in0_data_field0_field7(axis_sink[1].tdata[7*64+:64]),
    .in0_data_field1(axis_sink[1].tlast),
    
    // Src
    .out0_valid(axis_s0.tvalid),
    .out0_ready(axis_s0.tready),
    .out0_data_field0(axis_s0.tdata[0*64+:64]),
    .out0_data_field1(axis_s0.tlast)
);

// --------------------------------------------------------------------------------------
// Width conversions
// --------------------------------------------------------------------------------------
axis_dwidth_query_128_512 inst_dwidth_sender_query_0 (
    .aclk(aclk),
    .aresetn(aresetn),
    .s_axis_tvalid(axis_r0.tvalid),
    .s_axis_tready(axis_r0.tready),
    .s_axis_tdata (axis_r0.tdata),
    .s_axis_tlast (axis_r0.tlast),
    .m_axis_tvalid(axis_r1.tvalid),
    .m_axis_tready(axis_r1.tready),
    .m_axis_tdata (axis_r1.tdata),
    .m_axis_tlast (axis_r1.tlast),
    .m_axis_tkeep ()
);

// --------------------------------------------------------------------------------------
// Slave
// --------------------------------------------------------------------------------------
metaIntf #(.STYPE(req_t)) rs_req [2] ();
metaIntf #(.STYPE(req_t)) t_req ();
logic clear_hash;

join_slave inst_join_slave_0 (
    .aclk(aclk),
    .aresetn(aresetn),
    .axi_ctrl(axi_ctrl),
    .rs_req_out(rs_req),
    .t_req_out(t_req),
    .clear_hash(clear_hash)
);

// --------------------------------------------------------------------------------------
// Sender
// --------------------------------------------------------------------------------------
bpss_sender inst_sender (
    .aclk(aclk),
    .aresetn(aresetn),
    .t_req(t_req),
    .bpss_wr_req(bpss_wr_req),
    .axis_sink(axis_r1),
    .axis_src(axis_src)
);

// --------------------------------------------------------------------------------------
// Join
// --------------------------------------------------------------------------------------
join_top inst_join (
    .aclk(aclk),
    .aresetn(aresetn),
    .s_axis(axis_s0),
    .s_meta(s_join),
    .m_meta(m_join)
);

// --------------------------------------------------------------------------------------
// Hash
// --------------------------------------------------------------------------------------
distinct_top inst_distinct (
    .aclk(aclk),
    .aresetn(aresetn),
    .s_meta(s_hash),
    .m_meta(m_hash),
    .clear(clear_hash)
);

// --------------------------------------------------------------------------------------
// Requests
// --------------------------------------------------------------------------------------
user_bpss_rd inst_bpss_rd (
    .aclk(aclk),
    .aresetn(aresetn),
    .s_req(rs_req),
    .m_req(bpss_rd_req),
    .s_axis(axis_host_sink),
    .m_axis(axis_sink)
);

endmodule

