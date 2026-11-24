      bars[K_READY+kr.slot(t)].wait(kr.phase(t));
      qk(s,q,reinterpret_cast<half_t*>(arena+K_OFFSET+kr.slot(t)*TILE_BYTES),tid);
      if constexpr (CAUSAL) {
        if (t==diagonal_tile) {
          #pragma unroll
          for (int i=0;i<64;++i) {
            int row=(tid/32)*16+((tid&31)/4)+((i/2)%2)*8;
            int col=(i/4)*8+(tid%4)*2+i%2;
            if (col>row+int(fa3_query_block()%(128/QUERY_ROWS))*QUERY_ROWS) s[i]=-CUDART_INF_F;
          }
        }
      }
      // A TMA load zero-fills the tensor-map fringe. Padding logits must
      // receive zero probability, including on non-causal query rows.
      if constexpr (dynamic_geometry || (KV_ROWS % 128 != 0 && (!CAUSAL || ROWS>KV_ROWS))) {
        if ((nk%128)!=0 && t==tiles-1) {
          #pragma unroll
          for (int i=0;i<64;++i) {
            int col=(i/4)*8+(tid%4)*2+i%2;
            if (col>=nk%128) s[i]=-CUDART_INF_F;
          }
        }
      }
