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
always_comb bpss_wr_req.tie_off_m();
always_comb bpss_rd_done.tie_off_s();
always_comb bpss_wr_done.tie_off_s();

/* -- USER LOGIC -------------------------------------------------------- */
AXI4SR axis_sink ();
AXI4SR axis_src ();

logic [63:0] max_data;
logic max_last;
logic max_valid;

`AXISR_ASSIGN(axis_host_sink, axis_sink)
`AXISR_ASSIGN(axis_src, axis_host_src)

top_config_{{c}}_0 inst_top_{{c}} (
    .clock(aclk),
    .reset(~aresetn),
    
    // Sink
    .in0_valid(axis_sink.tvalid),
    .in0_ready(axis_sink.tready),
    .in0_data_field0_field0(axis_sink.tdata[0*64+:64]),
    .in0_data_field0_field1(axis_sink.tdata[1*64+:64]),
    .in0_data_field0_field2(axis_sink.tdata[2*64+:64]),
    .in0_data_field0_field3(axis_sink.tdata[3*64+:64]),
    .in0_data_field0_field4(axis_sink.tdata[4*64+:64]),
    .in0_data_field0_field5(axis_sink.tdata[5*64+:64]),
    .in0_data_field0_field6(axis_sink.tdata[6*64+:64]),
    .in0_data_field0_field7(axis_sink.tdata[7*64+:64]),
    .in0_data_field1(axis_sink.tlast),
    
    // Src
    .out0_valid(axis_src.tvalid),
    .out0_ready(axis_src.tready),
    .out0_data_field0_field0(axis_src.tdata[0*64+:64]),
    .out0_data_field0_field1(axis_src.tdata[1*64+:64]),
    .out0_data_field0_field2(axis_src.tdata[2*64+:64]),
    .out0_data_field0_field3(axis_src.tdata[3*64+:64]),
    .out0_data_field0_field4(axis_src.tdata[4*64+:64]),
    .out0_data_field0_field5(axis_src.tdata[5*64+:64]),
    .out0_data_field0_field6(axis_src.tdata[6*64+:64]),
    .out0_data_field0_field7(axis_src.tdata[7*64+:64]),
    .out0_data_field1(axis_src.tlast),
    
    .out1_valid(max_valid),
    .out1_ready(1'b1),
    
    .out1_data_field0(max_data),
    .out1_data_field1(max_last)
);

assign axis_src.tkeep = ~0;

//
// DEBUG
//
`ifdef DBG_PROBES_C0_{{c}}

    ila_stats_0 inst_ila_stats_{{c}} (
        .clk(aclk),
        
        .probe0(axis_sink.tvalid),
        .probe1(axis_sink.tready),
        .probe2(axis_sink.tlast),
        .probe3(axis_sink.tdata), // 512
        
        .probe4(axis_src.tvalid),
        .probe5(axis_src.tready),
        .probe6(axis_src.tlast),
        .probe7(axis_src.tdata), // 512
        
        .probe8(max_valid),
        .probe9(max_last),
        .probe10(max_data) // 64
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

    vio_stats_0 inst_vio_stats_{{c}} (
        .clk(aclk),
        .probe_in0(cnt_sink), // 32
        .probe_in1(cnt_src) // 32
    );

`endif


endmodule
