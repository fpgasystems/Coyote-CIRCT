import lynxTypes::*;
import aggTypes::*;

module distinct_hash #(
    parameter integer               AGG_DATA_BITS = AGG_KEY_BITS
) (
    input  logic                    aclk,
    input  logic                    aresetn,

    metaIntf.s                      s_lup_req,
    metaIntf.m                      m_lup_rsp,
    metaIntf.s                      s_upd_req
);

// Internal
logic [AGG_ADDR_BITS-1:0] hash_lup = 0;
logic [AGG_KEY_BITS-1:0] key_s0 = 0;
logic [AGG_KEY_BITS-1:0] key_s1 = 0;
logic last_s0 = 0;
logic last_s1 = 0;
logic val_s0 = 0;
logic val_s1 = 0;

logic [AGG_ADDR_BITS-1:0] hash_f_lup;
logic [AGG_DATA_BITS-1:0] data_lup;
logic entry_val;

// Updates
logic [AGG_ADDR_BITS-1:0] hash_upd = 0;
logic [AGG_ADDR_BITS-1:0] hash_f_upd;
logic [AGG_KEY_BITS-1:0] key_upd = 0;
logic val_upd = 0;


// Entry valid 
logic [2**AGG_ADDR_BITS-1:0] val_C = 0;
    
always_ff @( posedge aclk ) begin : REG
    if(~aresetn) begin
        val_C <= 0;

        hash_lup <= 0;
        key_s0 <= 0;
        key_s1 <= 0;
        val_s0 <= 0;
        val_s1 <= 0;
        last_s0 <= 0;
        last_s1 <= 0;

        hash_upd <= 0;
        key_upd <= 0;
        val_upd <= 0;

        m_lup_rsp.data <= 0;
        m_lup_rsp.valid <= 0;
    end
    else begin  
        // Valid entries
        if(last_s0) begin
            val_C <= 0;
        end

        // Updates
        hash_upd <= hash_f_upd;
        key_upd <= s_upd_req.data.key;
        val_upd <= s_upd_req.valid;
        val_C[hash_upd] <= val_upd ? 1'b1 : val_C[hash_upd];

        if(~stall) begin
            // S0
            hash_lup <= hash_f_lup;
            key_s0 <= s_lup_req.data.key;
            last_s0 <= s_lup_req.data.last;
            val_s0 <= s_lup_req.valid;
        
            // S1
            entry_val <= val_C[hash_lup];
            key_s1 <= key_s0;
            last_s1 <= last_s0;
            val_s1 <= val_s0;

            // S2
            m_lup_rsp.data.key <= data_lup;
            m_lup_rsp.data.hit <= (key_s1 == data_lup) && entry_val;
            m_lup_rsp.data.last <= last_s1;
            m_lup_rsp.valid <= val_s1;
        end
    end 
end

assign stall = m_lup_rsp.valid & ~m_lup_rsp.ready;
assign s_lup_req.ready = ~stall;

// Hash
hash #(
    .N_TABLES(1),
    .N_BLOCKS(2),
    .KEY_SIZE(AGG_KEY_BITS),
    .TABLE_SIZE(AGG_ADDR_BITS)
) inst_lup (
    .key_in(s_lup_req.data.key),
    .hash_out(hash_f_lup)
);

hash #(
    .N_TABLES(1),
    .N_BLOCKS(2),
    .KEY_SIZE(AGG_KEY_BITS),
    .TABLE_SIZE(AGG_ADDR_BITS)
) inst_upd (
    .key_in(s_upd_req.data.key),
    .hash_out(hash_f_upd)
);

// Hash table
logic [AGG_DATA_BITS/8-1:0] a_we;

assign a_we = {AGG_DATA_BITS/8{val_upd}};
assign s_upd_req.ready = 1'b1;

ram_tp_nc #(
    .ADDR_BITS(AGG_ADDR_BITS),
    .DATA_BITS(AGG_DATA_BITS)
) inst_hash (
    .clk(aclk),
    .a_en(1'b1),
    .a_we(a_we),
    .a_addr(hash_upd),
    .b_en(~stall),
    .b_addr(hash_lup),
    .a_data_in(key_upd),
    .a_data_out(),
    .b_data_out(data_lup)
);

endmodule