import lynxTypes::*;
import aggTypes::*;

module join_top (
    input  logic                    aclk,
    input  logic                    aresetn,

    AXI4SR.s                        s_axis,
    metaIntf.s                      s_meta,
    metaIntf.m                      m_meta
);

AXI4SR axis_q ();

// Queue
axis_data_fifo_query_64 inst_data_fifo_query (
    .s_axis_aclk(aclk),
    .s_axis_aresetn(aresetn),
    .s_axis_tvalid(s_axis.tvalid),
    .s_axis_tready(s_axis.tready),
    .s_axis_tdata (s_axis.tdata),
    .s_axis_tlast (s_axis.tlast),
    .m_axis_tvalid(axis_q.tvalid),
    .m_axis_tready(axis_q.tready),
    .m_axis_tdata (axis_q.tdata),
    .m_axis_tlast (axis_q.tlast)
);

// DP
always_comb begin: DP
    axis_q.tready = 1'b0;
    s_meta.ready = 1'b0;
    m_meta.valid = 1'b0;

    if(axis_q.tvalid && axis_q.tlast) begin
        axis_q.tready = 1'b1;
    end
    else begin
        if(axis_q.tvalid) begin
            m_meta.valid = s_meta.valid;
            m_meta.data.hit = s_meta.data.key == axis_q.tdata;
            s_meta.ready = 1'b1;

            axis_q.tready = s_meta.data.last;
        end
    end
end


endmodule