import lynxTypes::*;

module join_slave (
  input  logic                  aclk,
  input  logic                  aresetn,
  
  AXI4L.s                       axi_ctrl,

  metaIntf.m                    rs_req_out [2],
  metaIntf.m                    t_req_out,
  output logic                  clear_hash
);

// -- Decl ----------------------------------------------------------
// ------------------------------------------------------------------

// Constants
localparam integer N_REGS = 11;

localparam integer ADDR_LSB = $clog2(AXIL_DATA_BITS/8);
localparam integer ADDR_MSB = $clog2(N_REGS);
localparam integer AXI_ADDR_BITS = ADDR_LSB + ADDR_MSB;

// Internal registers
logic [AXI_ADDR_BITS-1:0] axi_awaddr;
logic axi_awready;
logic [AXI_ADDR_BITS-1:0] axi_araddr;
logic axi_arready;
logic [1:0] axi_bresp;
logic axi_bvalid;
logic axi_wready;
logic [AXIL_DATA_BITS-1:0] axi_rdata;
logic [1:0] axi_rresp;
logic axi_rvalid;

// Registers
logic [N_REGS-1:0][AXIL_DATA_BITS-1:0] slv_reg;
logic slv_reg_rden;
logic slv_reg_wren;
logic aw_en;

logic [31:0] used_queue;
logic [LEN_BITS-1:0] s_tuples;

// -- Def -----------------------------------------------------------
// ------------------------------------------------------------------

// -- Register map ----------------------------------------------------------------------- 
localparam integer CTRL_REG = 0;
localparam integer STAT_REG = 2;
localparam integer VADDR_REG_R = 2;
localparam integer VADDR_REG_S = 3;
localparam integer VADDR_REG_T = 4;
localparam integer PID_REG_R = 5;
localparam integer PID_REG_S = 6;
localparam integer PID_REG_T = 7;
localparam integer SIZE_REG_R = 8;
localparam integer SIZE_REG_S = 9;
localparam integer SIZE_REG_T = 10;

// Write process
assign slv_reg_wren = axi_wready && axi_ctrl.wvalid && axi_awready && axi_ctrl.awvalid;

always_ff @(posedge aclk) begin
  if ( aresetn == 1'b0 ) begin
    slv_reg <= 0;
  end
  else begin
    slv_reg[CTRL_REG] <= 0;

    if(slv_reg_wren) begin
      case (axi_awaddr[ADDR_LSB+:ADDR_MSB])
        CTRL_REG: // CTRL
          for (int i = 0; i < (AXIL_DATA_BITS/8); i++) begin
            if(axi_ctrl.wstrb[i]) begin
              slv_reg[CTRL_REG][(i*8)+:8] <= axi_ctrl.wdata[(i*8)+:8];
            end
          end
        VADDR_REG_R: // VADDR
          for (int i = 0; i < (AXIL_DATA_BITS/8); i++) begin
            if(axi_ctrl.wstrb[i]) begin
              slv_reg[VADDR_REG_R][(i*8)+:8] <= axi_ctrl.wdata[(i*8)+:8];
            end
          end
        VADDR_REG_S: // VADDR
          for (int i = 0; i < (AXIL_DATA_BITS/8); i++) begin
            if(axi_ctrl.wstrb[i]) begin
              slv_reg[VADDR_REG_S][(i*8)+:8] <= axi_ctrl.wdata[(i*8)+:8];
            end
          end
        VADDR_REG_T: // VADDR
          for (int i = 0; i < (AXIL_DATA_BITS/8); i++) begin
            if(axi_ctrl.wstrb[i]) begin
              slv_reg[VADDR_REG_T][(i*8)+:8] <= axi_ctrl.wdata[(i*8)+:8];
            end
          end
        PID_REG_R: // PID
          for (int i = 0; i < (AXIL_DATA_BITS/8); i++) begin
            if(axi_ctrl.wstrb[i]) begin
              slv_reg[PID_REG_R][(i*8)+:8] <= axi_ctrl.wdata[(i*8)+:8];
            end
          end
        PID_REG_S: // PID
          for (int i = 0; i < (AXIL_DATA_BITS/8); i++) begin
            if(axi_ctrl.wstrb[i]) begin
              slv_reg[PID_REG_S][(i*8)+:8] <= axi_ctrl.wdata[(i*8)+:8];
            end
          end
        PID_REG_T: // PID
          for (int i = 0; i < (AXIL_DATA_BITS/8); i++) begin
            if(axi_ctrl.wstrb[i]) begin
              slv_reg[PID_REG_T][(i*8)+:8] <= axi_ctrl.wdata[(i*8)+:8];
            end
          end
        SIZE_REG_R: // PID
          for (int i = 0; i < (AXIL_DATA_BITS/8); i++) begin
            if(axi_ctrl.wstrb[i]) begin
              slv_reg[SIZE_REG_R][(i*8)+:8] <= axi_ctrl.wdata[(i*8)+:8];
            end
          end
        SIZE_REG_S: // PID
          for (int i = 0; i < (AXIL_DATA_BITS/8); i++) begin
            if(axi_ctrl.wstrb[i]) begin
              slv_reg[SIZE_REG_S][(i*8)+:8] <= axi_ctrl.wdata[(i*8)+:8];
            end
          end
        SIZE_REG_T: // PID
          for (int i = 0; i < (AXIL_DATA_BITS/8); i++) begin
            if(axi_ctrl.wstrb[i]) begin
              slv_reg[SIZE_REG_T][(i*8)+:8] <= axi_ctrl.wdata[(i*8)+:8];
            end
          end
        default : ;
      endcase
    end
  end
end    

// Read process
assign slv_reg_rden = axi_arready & axi_ctrl.arvalid & ~axi_rvalid;

always_ff @(posedge aclk) begin
  if( aresetn == 1'b0 ) begin
    axi_rdata <= 0;
  end
  else begin
    if(slv_reg_rden) begin
      axi_rdata <= 0;
      case (axi_araddr[ADDR_LSB+:ADDR_MSB])
        STAT_REG: // STAT
          axi_rdata[31:0] <= used_queue;
        default: ;
      endcase
    end
  end 
end

// Params
localparam integer PMTU_BEATS = PMTU_BYTES / (AXI_DATA_BITS / 8);
localparam integer BEAT_LOG_BITS = $clog2(AXI_DATA_BITS / 8);

// FSM
typedef enum logic[1:0]  {ST_IDLE, ST_SEND_R} state_t;
logic [1:0] state_C = ST_IDLE, state_N;

logic [LEN_BITS-1:0] cnt_C, cnt_N;

metaIntf #(.STYPE(req_t)) cmd_req ();
metaIntf #(.STYPE(req_t)) in_req ();
metaIntf #(.STYPE(req_t)) t_req ();
metaIntf #(.STYPE(req_t)) r_req ();
metaIntf #(.STYPE(req_t)) s_req ();

assign cmd_req.valid = slv_reg[CTRL_REG][0];

axis_data_fifo_req_96_used inst_cmd_queue_rd (
  .s_axis_aresetn(aresetn),
  .s_axis_aclk(aclk),
  .s_axis_tvalid(cmd_req.valid),
  .s_axis_tready(cmd_req.ready),
  .s_axis_tdata(cmd_req.data),
  .m_axis_tvalid(in_req.valid),
  .m_axis_tready(in_req.ready),
  .m_axis_tdata(in_req.data),
  .axis_wr_data_count(used_queue)
);

queue_meta #(.QDEPTH(8)) inst_t (.aclk(aclk), .aresetn(aresetn), .s_meta(t_req), .m_meta(t_req_out));
queue_meta #(.QDEPTH(8)) inst_s (.aclk(aclk), .aresetn(aresetn), .s_meta(s_req), .m_meta(rs_req_out[0]));
queue_meta #(.QDEPTH(8)) inst_r (.aclk(aclk), .aresetn(aresetn), .s_meta(r_req), .m_meta(rs_req_out[1]));

// REG
always_ff @(posedge aclk) begin: PROC_REG
if (aresetn == 1'b0) begin
    state_C <= ST_IDLE;
    cnt_C <= 0;
end
else
    state_C <= state_N;
    cnt_C <= cnt_N;
end

// NSL
always_comb begin: NSL
    state_N = state_C;

    case(state_C) 
        ST_IDLE: 
            if(in_req.valid & t_req.ready & s_req.ready)    
                state_N = ST_SEND_R;
            
        ST_SEND_R:
            if(r_req.ready && (cnt_C == s_tuples))
                state_N = ST_IDLE;

    endcase
end

// DP
always_comb begin: DP
    cnt_N = cnt_C;
   
    s_tuples = (slv_reg[SIZE_REG_S] - 1) >> BEAT_LOG_BITS;

    in_req.ready = 1'b0;

    t_req.valid = 1'b0;
    t_req.data.vaddr = slv_reg[VADDR_REG_T];
    t_req.data.len = slv_reg[SIZE_REG_T];
    t_req.data.pid = slv_reg[PID_REG_T];
    t_req.data.ctl = 1'b1;

    s_req.valid = 1'b0;
    s_req.data.vaddr = slv_reg[VADDR_REG_S];
    s_req.data.len = slv_reg[SIZE_REG_S];
    s_req.data.pid = slv_reg[PID_REG_S];
    s_req.data.ctl = 1'b1;

    r_req.valid = 1'b0;
    r_req.data.vaddr = slv_reg[VADDR_REG_R];
    r_req.data.len = slv_reg[SIZE_REG_R];
    r_req.data.pid = slv_reg[PID_REG_R];
    r_req.data.ctl = 1'b1;

    clear_hash = 1'b0;

    case(state_C)
        ST_IDLE: begin
            cnt_N = 0;
            if(in_req.valid & t_req.ready & s_req.ready) begin
                in_req.ready = 1'b1;
                t_req.valid = 1'b1;
                s_req.valid = 1'b1;
                clear_hash = 1'b1;
            end
        end

        ST_SEND_R: begin
            r_req.valid = 1'b1;

            if(r_req.ready) begin
                if(cnt_C == s_tuples)
                    cnt_N = 0;
                else
                    cnt_N = cnt_C + 1;
            end
        end

    endcase
end

// I/O
assign axi_ctrl.awready = axi_awready;
assign axi_ctrl.arready = axi_arready;
assign axi_ctrl.bresp = axi_bresp;
assign axi_ctrl.bvalid = axi_bvalid;
assign axi_ctrl.wready = axi_wready;
assign axi_ctrl.rdata = axi_rdata;
assign axi_ctrl.rresp = axi_rresp;
assign axi_ctrl.rvalid = axi_rvalid;

// awready and awaddr
always_ff @(posedge aclk) begin
  if ( aresetn == 1'b0 )
    begin
      axi_awready <= 1'b0;
      axi_awaddr <= 0;
      aw_en <= 1'b1;
    end 
  else
    begin    
      if (~axi_awready && axi_ctrl.awvalid && axi_ctrl.wvalid && aw_en)
        begin
          axi_awready <= 1'b1;
          aw_en <= 1'b0;
          axi_awaddr <= axi_ctrl.awaddr;
        end
      else if (axi_ctrl.bready && axi_bvalid)
        begin
          aw_en <= 1'b1;
          axi_awready <= 1'b0;
        end
      else           
        begin
          axi_awready <= 1'b0;
        end
    end 
end  

// arready and araddr
always_ff @(posedge aclk) begin
  if ( aresetn == 1'b0 )
    begin
      axi_arready <= 1'b0;
      axi_araddr  <= 0;
    end 
  else
    begin    
      if (~axi_arready && axi_ctrl.arvalid)
        begin
          axi_arready <= 1'b1;
          axi_araddr  <= axi_ctrl.araddr;
        end
      else
        begin
          axi_arready <= 1'b0;
        end
    end 
end    

// bvalid and bresp
always_ff @(posedge aclk) begin
  if ( aresetn == 1'b0 )
    begin
      axi_bvalid  <= 0;
      axi_bresp   <= 2'b0;
    end 
  else
    begin    
      if (axi_awready && axi_ctrl.awvalid && ~axi_bvalid && axi_wready && axi_ctrl.wvalid)
        begin
          axi_bvalid <= 1'b1;
          axi_bresp  <= 2'b0;
        end                   
      else
        begin
          if (axi_ctrl.bready && axi_bvalid) 
            begin
              axi_bvalid <= 1'b0; 
            end  
        end
    end
end

// wready
always_ff @(posedge aclk) begin
  if ( aresetn == 1'b0 )
    begin
      axi_wready <= 1'b0;
    end 
  else
    begin    
      if (~axi_wready && axi_ctrl.wvalid && axi_ctrl.awvalid && aw_en )
        begin
          axi_wready <= 1'b1;
        end
      else
        begin
          axi_wready <= 1'b0;
        end
    end 
end  

// rvalid and rresp (1Del?)
always_ff @(posedge aclk) begin
  if ( aresetn == 1'b0 )
    begin
      axi_rvalid <= 0;
      axi_rresp  <= 0;
    end 
  else
    begin    
      if (axi_arready && axi_ctrl.arvalid && ~axi_rvalid)
        begin
          axi_rvalid <= 1'b1;
          axi_rresp  <= 2'b0;
        end   
      else if (axi_rvalid && axi_ctrl.rready)
        begin
          axi_rvalid <= 1'b0;
        end                
    end
end    

endmodule // cnfg_slave