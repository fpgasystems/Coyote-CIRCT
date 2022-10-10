module top(	// <stdin>:1103:3
  input         in0_valid,
  input  [63:0] in0_data_field0,
                in0_data_field1,
                in0_data_field2,
                in0_data_field3,
                in0_data_field4,
                in0_data_field5,
                in0_data_field6,
                in0_data_field7,
  input         inCtrl_valid,
                out0_ready,
                outCtrl_ready,
                clock,
                reset,
  output        in0_ready,
                inCtrl_ready,
                out0_valid,
  output [63:0] out0_data_field0,
                out0_data_field1,
                out0_data_field2,
                out0_data_field3,
                out0_data_field4,
                out0_data_field5,
                out0_data_field6,
                out0_data_field7,
  output        outCtrl_valid);

assign out0_data_field0 = in0_data_field0;
assign out0_data_field1 = in0_data_field1;
assign out0_data_field2 = in0_data_field2;
assign out0_data_field3 = in0_data_field3;
assign out0_data_field4 = in0_data_field4;
assign out0_data_field5 = in0_data_field5;
assign out0_data_field6 = in0_data_field6;
assign out0_data_field7 = in0_data_field7;

assign out0_valid = in0_valid;
assign in0_ready = out0_ready;

assign inCtrl_ready = 1'b1;
assign outCtrl_valid = 1'b1;

endmodule

