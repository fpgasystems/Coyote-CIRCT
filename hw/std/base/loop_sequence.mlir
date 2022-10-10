func.func @compute(%val: i64) -> i64 {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c3 = arith.constant 3 : i64
  %c16 = arith.constant 16 : i64
  %mask = arith.constant 4660 : i64 // 0x1234
  cf.br ^bb1(%c0, %val: i64, i64)
^bb1(%i0 : i64, %v0: i64):
  %cond0 = arith.cmpi ne, %i0, %c3 : i64
  cf.cond_br %cond0, ^bb2, ^bb3(%c0, %v0: i64, i64)
^bb2:
  %x0 = arith.xori %v0, %mask : i64
  %s0 = arith.shrui %v0, %c16 : i64
  %r0 = arith.xori %x0, %s0 : i64
  %in0 = arith.addi %i0, %c1 : i64
  cf.br ^bb1(%in0, %r0: i64, i64)
^bb3(%i1: i64, %v1: i64):
  %cond1 = arith.cmpi ne, %i1, %c3 : i64
  cf.cond_br %cond1, ^bb4, ^bb5(%c0, %v1: i64, i64)
^bb4:
  %x1 = arith.xori %v1, %mask : i64
  %s1 = arith.shrui %v1, %c16 : i64
  %r1 = arith.xori %x1, %s1 : i64
  %in1 = arith.addi %i1, %c1 : i64
  cf.br ^bb3(%in1, %r1: i64, i64)
^bb5(%i2: i64, %v2: i64):
  %cond2 = arith.cmpi ne, %i2, %c3 : i64
  cf.cond_br %cond2, ^bb6, ^bb7(%v2: i64)
^bb6:
  %x2 = arith.xori %v2, %mask : i64
  %s2 = arith.shrui %v2, %c16 : i64
  %r2 = arith.xori %x2, %s2 : i64
  %in2 = arith.addi %i2, %c1 : i64
  cf.br ^bb5(%in2, %r2: i64, i64)
^bb7(%res: i64):
  return %res: i64
}
