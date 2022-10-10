#
# Add IP cores
#
if {![info exists axis_data_fifo_sender_agg_512]} {
    create_ip -name axis_data_fifo -vendor xilinx.com -library ip -version 2.0 -module_name axis_data_fifo_sender_agg_512
    set_property -dict [list CONFIG.TDATA_NUM_BYTES {64} CONFIG.HAS_TLAST {1}] [get_ips axis_data_fifo_sender_agg_512]
    set axis_data_fifo_sender_agg_512 1
}

if {![info exists ila_agg_0]} {
    create_ip -name ila -vendor xilinx.com -library ip -version 6.2 -module_name ila_agg_0
    set_property -dict [list CONFIG.C_PROBE28_WIDTH {6} CONFIG.C_PROBE26_WIDTH {28} CONFIG.C_PROBE25_WIDTH {48} CONFIG.C_PROBE22_WIDTH {48} CONFIG.C_PROBE21_WIDTH {32} CONFIG.C_PROBE20_WIDTH {2} CONFIG.C_PROBE19_WIDTH {66} CONFIG.C_PROBE16_WIDTH {66} CONFIG.C_PROBE7_WIDTH {512} CONFIG.C_PROBE3_WIDTH {512} CONFIG.C_DATA_DEPTH {2048} CONFIG.C_NUM_OF_PROBES {29} CONFIG.Component_Name {ila_agg_0} CONFIG.C_EN_STRG_QUAL {1} CONFIG.C_ADV_TRIGGER {true} CONFIG.C_PROBE28_MU_CNT {2} CONFIG.C_PROBE27_MU_CNT {2} CONFIG.C_PROBE26_MU_CNT {2} CONFIG.C_PROBE25_MU_CNT {2} CONFIG.C_PROBE24_MU_CNT {2} CONFIG.C_PROBE23_MU_CNT {2} CONFIG.C_PROBE22_MU_CNT {2} CONFIG.C_PROBE21_MU_CNT {2} CONFIG.C_PROBE20_MU_CNT {2} CONFIG.C_PROBE19_MU_CNT {2} CONFIG.C_PROBE18_MU_CNT {2} CONFIG.C_PROBE17_MU_CNT {2} CONFIG.C_PROBE16_MU_CNT {2} CONFIG.C_PROBE15_MU_CNT {2} CONFIG.C_PROBE14_MU_CNT {2} CONFIG.C_PROBE13_MU_CNT {2} CONFIG.C_PROBE12_MU_CNT {2} CONFIG.C_PROBE11_MU_CNT {2} CONFIG.C_PROBE10_MU_CNT {2} CONFIG.C_PROBE9_MU_CNT {2} CONFIG.C_PROBE8_MU_CNT {2} CONFIG.C_PROBE7_MU_CNT {2} CONFIG.C_PROBE6_MU_CNT {2} CONFIG.C_PROBE5_MU_CNT {2} CONFIG.C_PROBE4_MU_CNT {2} CONFIG.C_PROBE3_MU_CNT {2} CONFIG.C_PROBE2_MU_CNT {2} CONFIG.C_PROBE1_MU_CNT {2} CONFIG.C_PROBE0_MU_CNT {2} CONFIG.ALL_PROBE_SAME_MU {true} CONFIG.ALL_PROBE_SAME_MU_CNT {2}] [get_ips ila_agg_0]
    set ila_agg_0 1
}

if {![info exists vio_agg_0]} {
    create_ip -name vio -vendor xilinx.com -library ip -version 3.0 -module_name vio_agg_0
    set_property -dict [list CONFIG.C_PROBE_IN1_WIDTH {32} CONFIG.C_PROBE_IN0_WIDTH {32} CONFIG.C_NUM_PROBE_OUT {0} CONFIG.C_NUM_PROBE_IN {2}] [get_ips vio_agg_0]
    set vio_agg_0 1
}


