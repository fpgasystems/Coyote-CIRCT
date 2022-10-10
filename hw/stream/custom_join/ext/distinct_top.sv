import lynxTypes::*;
import aggTypes::*;

module distinct_top (
    input  logic                aclk,
    input  logic                aresetn,

    metaIntf.s                  s_meta,
    metaIntf.m                  m_meta,
    input  logic                clear
);

metaIntf #(.STYPE(ext_t)) lup_cache ();
metaIntf #(.STYPE(ext_t)) lup_que_in ();
metaIntf #(.STYPE(ext_t)) lup_que_out ();
metaIntf #(.STYPE(ext_t)) lup_req ();
metaIntf #(.STYPE(ext_t)) lup_rsp ();
metaIntf #(.STYPE(ext_t)) upd_req ();
metaIntf #(.STYPE(ext_t)) que_out ();

// Cache
lru_cache #(
    .CACHE_DEPTH(CACHE_DEPTH)
) inst_lru (
    .aclk(aclk),
    .aresetn(aresetn),
    .s_meta(s_meta),
    .m_meta(lup_cache)
);

// Queue lup
queue_meta #(
    .QDEPTH(QDEPTH_DIST)
) inst_queue_lup (
    .aclk(aclk),
    .aresetn(aresetn),
    .s_meta(lup_que_in),
    .m_meta(lup_que_out)
);

// Hash
distinct_hash inst_hash (
    .aclk(aclk),
    .aresetn(aresetn),
    .s_lup_req(lup_req),
    .m_lup_rsp(lup_rsp),
    .s_upd_req(upd_req),
    .clear(clear)
);

// S0
always_comb begin : DP_S0
    lup_cache.ready = lup_que_in.ready & lup_req.ready;
    lup_que_in.valid = lup_cache.valid & lup_cache.ready;
    lup_req.valid = lup_cache.valid & lup_cache.ready;

    lup_que_in.data = lup_cache.data;
    lup_req.data = lup_cache.data;
end

// S1
always_comb begin : DP_S1
    // Source
    que_out.data.key = lup_que_out.data.key;
    que_out.data.last = lup_que_out.data.last;
    que_out.data.hit = lup_que_out.data.hit || lup_rsp.data.hit;
    que_out.valid = lup_rsp.valid;

    // Updates
    upd_req.data.key = lup_que_out.data.key;
    upd_req.valid = que_out.valid & que_out.ready;

    lup_que_out.ready = que_out.valid & que_out.ready;
    lup_rsp.ready = que_out.valid & que_out.ready;
end

// Queue out
queue_meta #(
    .QDEPTH(QDEPTH_DIST)
) inst_queue_out (
    .aclk(aclk),
    .aresetn(aresetn),
    .s_meta(que_out),
    .m_meta(m_meta)
);
    
endmodule