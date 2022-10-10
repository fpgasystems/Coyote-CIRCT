import lynxTypes::*;
import aggTypes::*;

module lru_cache #(
    parameter integer           CACHE_DEPTH = 8
) (
    input  logic                aclk,
    input  logic                aresetn,

    metaIntf.s                  s_meta,
    metaIntf.m                  m_meta
);
    
dist_t [CACHE_DEPTH-1:0] cache = 0;
logic [CACHE_DEPTH-1:0] valid = 0;

logic hit;
logic stall;

always_ff @( posedge aclk ) begin : REG
    if(~aresetn) begin
        cache <= 0;
        valid <= 0;
    end
    else begin
        if(~stall) begin
            cache[0].key <= s_meta.data.key;
            cache[0].last <= s_meta.data.last;
            cache[0].hit <= hit;
            valid[0] <= s_meta.valid;

            for(int i = 1; i < CACHE_DEPTH; i++) begin
                cache[i].key <= cache[i-1].key;
                cache[i].last <= cache[i-1].last;
                cache[i].hit <= cache[i-1].hit;
                valid[i] <= valid[i-1];
            end
        end
    end
end

always_comb begin : DP
    hit = 1'b0;

    for(int i = 0; i < CACHE_DEPTH-1; i++) begin
        if(valid[i]) begin
            if(s_meta.data.key == cache[i].key)
                hit = 1'b1;
        end
    end
end

assign m_meta.data.hit = cache[CACHE_DEPTH-1].hit;
assign m_meta.data.key = cache[CACHE_DEPTH-1].key;
assign m_meta.data.last = cache[CACHE_DEPTH-1].last;
assign m_meta.valid = valid[CACHE_DEPTH-1];

assign stall = m_meta.valid & ~m_meta.ready;
assign s_meta.ready = ~stall;

endmodule