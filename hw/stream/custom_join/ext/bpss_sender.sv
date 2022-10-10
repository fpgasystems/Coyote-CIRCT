module bpss_sender (
    // Descriptor sink
    metaIntf.s                                  t_req,

    // Descriptor source
    metaIntf.m			                        bpss_wr_req,

    // AXI4S HOST STREAMS
    AXI4SR.s                                    axis_sink,
    AXI4SR.m                                    axis_src,

    // Clock and reset
    input  wire                                 aclk,
    input  wire[0:0]                            aresetn
);

// Params
localparam integer PMTU_BEATS = PMTU_BYTES / (AXI_DATA_BITS / 8);
localparam integer BEAT_LOG_BITS = $clog2(AXI_DATA_BITS / 8);

// FSM
typedef enum logic[1:0]  {ST_IDLE, ST_FILL, ST_SEND, ST_SEND_LAST} state_t;
logic [1:0] state_C = ST_IDLE, state_N;

// Regs
logic [31:0] cnt_q_C, cnt_q_N;
logic [VADDR_BITS-1:0] addr_C = 0, addr_N;
logic [VADDR_BITS-1:0] vaddr_C = 0, vaddr_N;
logic [PID_BITS-1:0] pid_C = 0, pid_N;

// Internal
AXI4SR axis_s0 ();

// REG
always_ff @(posedge aclk) begin: PROC_REG
if (aresetn == 1'b0) begin
    state_C <= ST_IDLE;
    cnt_q_C <= 0;
    addr_C <= 0;
    vaddr_C <= 0;
    pid_C <= 0;
end
else
    state_C <= state_N;
    cnt_q_C <= cnt_q_N;
    addr_C <= addr_N;
    vaddr_C <= vaddr_N;
    pid_C <= pid_N;
end

// NSL
always_comb begin: NSL
    state_N = state_C;

    case(state_C) 
        ST_IDLE: 
            if(t_req.valid)
                state_N = ST_FILL;

        ST_FILL: 
            if(axis_sink.tvalid & axis_sink.tready) begin
                if(axis_sink.tlast) begin
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
    vaddr_N = vaddr_C;
    pid_N = pid_C;

    // Bypass
    bpss_wr_req.valid = 1'b0;
    bpss_wr_req.data = 0;
    bpss_wr_req.data.vaddr = vaddr_C + addr_C;
    bpss_wr_req.data.len =  cnt_q_C << BEAT_LOG_BITS;
    bpss_wr_req.data.ctl = 1'b0;
    bpss_wr_req.data.pid = pid_C;

    // Handshake
    axis_sink.tready = 1'b0;
    axis_s0.tvalid = 1'b0;
    axis_s0.tdata = axis_sink.tdata;
    axis_s0.tlast = axis_sink.tlast;

    t_req.ready = 1'b0;

    case(state_C) 
        ST_IDLE: begin
            t_req.ready = 1'b1;
            if(t_req.valid) begin
                vaddr_N = t_req.data.vaddr;
                pid_N = t_req.data.pid;
            end
        end

        ST_FILL: begin
            axis_s0.tvalid = axis_sink.tvalid;
            axis_sink.tready = axis_s0.tready;

            if(axis_sink.tvalid & axis_sink.tready) begin
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

// Queue out
axis_data_fifo_sender_agg_512 inst_data_fifo_sender_agg_0 (
    .s_axis_aclk(aclk),
    .s_axis_aresetn(aresetn),
    .s_axis_tvalid(axis_s0.tvalid),
    .s_axis_tready(axis_s0.tready),
    .s_axis_tdata (axis_s0.tdata),
    .s_axis_tlast (axis_s0.tlast),
    .m_axis_tvalid(axis_src.tvalid),
    .m_axis_tready(axis_src.tready),
    .m_axis_tdata (axis_src.tdata),
    .m_axis_tlast (axis_src.tlast)
);
assign axis_src.tkeep = ~0;
    
endmodule