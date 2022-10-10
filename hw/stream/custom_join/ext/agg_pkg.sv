package aggTypes;

     parameter integer QDEPTH_DIST = 8;
     parameter integer CACHE_DEPTH = 8;

     parameter integer AGG_KEY_BITS = 64;
     parameter integer AGG_ADDR_BITS = 10;

     typedef struct packed {
          logic [AGG_KEY_BITS-1:0] key;
          logic hit;
          logic last;
     } ext_t;
    
endpackage