#
# Add IP cores
#
if {![info exists ila_base_0]} {
    create_ip -name ila -vendor xilinx.com -library ip -version 6.2 -module_name ila_base_0
    set_property -dict [list CONFIG.C_PROBE6_WIDTH {512} CONFIG.C_PROBE3_WIDTH {512} CONFIG.C_DATA_DEPTH {2048} CONFIG.C_NUM_OF_PROBES {7} CONFIG.C_EN_STRG_QUAL {1} CONFIG.C_ADV_TRIGGER {true} CONFIG.C_PROBE6_MU_CNT {2} CONFIG.C_PROBE5_MU_CNT {2} CONFIG.C_PROBE4_MU_CNT {2} CONFIG.C_PROBE3_MU_CNT {2} CONFIG.C_PROBE2_MU_CNT {2} CONFIG.C_PROBE1_MU_CNT {2} CONFIG.C_PROBE0_MU_CNT {2} CONFIG.ALL_PROBE_SAME_MU_CNT {2}] [get_ips ila_base_0]    
    set ila_base_0 1
}

if {![info exists vio_base_0]} {
    create_ip -name vio -vendor xilinx.com -library ip -version 3.0 -module_name vio_base_0
    set_property -dict [list CONFIG.C_PROBE_IN1_WIDTH {32} CONFIG.C_PROBE_IN0_WIDTH {32} CONFIG.C_NUM_PROBE_OUT {0} CONFIG.C_NUM_PROBE_IN {2}] [get_ips vio_base_0]
    set vio_base_0 1
}


