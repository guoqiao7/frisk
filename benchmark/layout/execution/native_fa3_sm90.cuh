// SM90 FA3 replay. Constants in the companion translation unit come from
// the Frisk execution contract and the two verified MMA descriptor plans.
#include <tl_templates/cuda/instruction/wgmma.h>
#include <tl_templates/cuda/intrin.h>
#include <tl_templates/cuda/barrier.h>
#include <tl_templates/cuda/copy_sm90.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <math_constants.h>

#ifndef FRISK_KV_HEAD_RATIO
#define FRISK_KV_HEAD_RATIO 1
#endif
// Descending causal work, with heads interleaved at each query coordinate.
__device__ __forceinline__ int fa3_query_block() {
  if constexpr (REVERSE_Q) return gridDim.x-1-(blockIdx.x+gridDim.x*blockIdx.y)/gridDim.y;
  return blockIdx.x;
}
__device__ __forceinline__ int fa3_head() {
  if constexpr (REVERSE_Q) return (blockIdx.x+gridDim.x*blockIdx.y)%gridDim.y;
  return blockIdx.y;
}
struct Ring {
  int versions, ready, free_slot, byte_offset, issuer;
  __device__ int slot(int i) const { return i % versions; }
  __device__ int phase(int i) const { return (i / versions) & 1; }
  template<int Rows=MAP_ROWS, int HeadRatio=FRISK_KV_HEAD_RATIO>
  __device__ void load(const CUtensorMap& map, int tile, int row, int local,
                       unsigned char* arena, Barrier* bars) const {
    if (free_slot >= 0 && tile >= versions)
      bars[free_slot+slot(tile)].wait(phase(tile-versions));
    if (local == issuer) {
      auto& b=bars[ready+slot(tile)];
      constexpr int bytes=Rows*MAP_COLS*4;
      b.arrive_and_expect_tx(bytes);
      auto* dst=reinterpret_cast<half_t*>(arena+byte_offset+slot(tile)*bytes);
      tl::tma_load(map,b,dst,0,row,int(fa3_head()/HeadRatio),0);
      tl::tma_load(map,b,dst+Rows*MAP_COLS,int(MAP_COLS),row,int(fa3_head()/HeadRatio),0);
    }
  }
};

__device__ __forceinline__ void qk(float (&s)[64], const half_t* q, const half_t* k, int tid) {
  tl::GmmaDescriptor a,b;
  tl::initialize_wgmma_descriptor<SS_SW,QA_LEAD,QA_STRIDE>(a,q);
  tl::initialize_wgmma_descriptor<SS_SW,KB_LEAD,KB_STRIDE>(b,k);
  #pragma unroll
  for (int i=0;i<64;++i) s[i]=0;
  tl::warpgroup_fence_operand(s,64);
  tl::warpgroup_arrive();
  #pragma unroll
  for (int i=0;i<BATCH;++i) {
    auto ao=QK_A_START[(tid/128)*BATCH+i], bo=QK_B_START[i];
    tl::wgmma_ss<tl::DataType::kFloat16,tl::DataType::kFloat16,tl::DataType::kFloat32,
                 64,128,16,false,false,1,1>(uint64_t(a+(ao>>4)),uint64_t(b+(bo>>4)),
                                             reinterpret_cast<uint32_t*>(s),1);
  }
  tl::warpgroup_commit_batch();
  tl::warpgroup_wait<WAIT>();
  tl::warpgroup_fence_operand(s,64);
}

__device__ __forceinline__ void pv(float (&o)[64], half_t (&p)[64], const half_t* v) {
  tl::GmmaDescriptor b;
  tl::initialize_wgmma_descriptor<PV_SW,VB_LEAD,VB_STRIDE>(b,v);
  auto* packed=reinterpret_cast<uint32_t*>(p);
  tl::warpgroup_fence_operand(packed,32);
  tl::warpgroup_fence_operand(o,64);
  tl::warpgroup_arrive();
  #pragma unroll
  for (int i=0;i<BATCH;++i) {
    tl::wgmma_rs<tl::DataType::kFloat16,tl::DataType::kFloat16,tl::DataType::kFloat32,
                 64,128,16,false,true,1,1>(packed+i*4,uint64_t(b+(PV_B_START[i]>>4)),
                                           reinterpret_cast<uint32_t*>(o),1);
  }
  tl::warpgroup_commit_batch();
  tl::warpgroup_wait<WAIT>();
  tl::warpgroup_fence_operand(o,64);
  tl::warpgroup_fence_operand(packed,32);
}

__device__ __forceinline__ float attention_exp2(float x) {
#ifdef FRISK_EXP_FTZ
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
#else
  return exp2f(x);
#endif
}

// Each lane owns two rows and 32 columns per row in a m64n128 accumulator.
__device__ __forceinline__ void update(float (&s)[64], half_t (&p)[64],
    float (&o)[64], float (&maxima)[2], float (&sums)[2]) {
  constexpr float scale=0x1.0527dbd5cafffp-3f;
  #pragma unroll
  for (int row=0;row<2;++row) {
    float m=-CUDART_INF_F;
    #pragma unroll
    for (int j=0;j<32;++j) m=fmaxf(m,s[(j%16)*4+row*2+j/16]);
    m=tl::AllReduce<tl::MaxOp,4,1,0,tl::NamedBarrier<C_THREADS>>::run(m);
    m=fmaxf(m,maxima[row]);
    float factor=attention_exp2(maxima[row]*scale-m*scale), sum=0;
    #pragma unroll
    for (int j=0;j<32;++j) {
      int idx=(j%16)*4+row*2+j/16;
      float e=attention_exp2(s[idx]*scale-m*scale);
      sum+=e; p[idx]=half_t(e); o[idx]*=factor;
    }
    sum=tl::AllReduce<tl::SumOp,4,1,0,tl::NamedBarrier<C_THREADS>>::run(sum);
    maxima[row]=m; sums[row]=sums[row]*factor+sum;
  }
}

#include "native_fa3_shortkv.cuh"

struct Geometry {
  int queries, keys, output_pitch, qp, kp, vp;
  const half_t *q, *k, *v;
};
struct CtaTrace { unsigned long long start, end; unsigned sm, iterations; };
#ifdef FRISK_DYNAMIC_GEOMETRY
constexpr bool dynamic_geometry = true;
#else
constexpr bool dynamic_geometry = false;
#endif

extern "C" __global__ void __launch_bounds__(CTA,1) main_kernel(
    __grid_constant__ const CUtensorMap K, half_t* output,
    __grid_constant__ const CUtensorMap Q, __grid_constant__ const CUtensorMap V, Geometry shape, CtaTrace* trace) {
  const int nq = dynamic_geometry ? shape.queries : ROWS;
  const int nk = dynamic_geometry ? shape.keys : KV_ROWS;
  const int tiles = (nk+127)/128;
#ifdef FRISK_TRACE_CTAS
  auto& record=trace[blockIdx.x+gridDim.x*blockIdx.y];
  if (threadIdx.x==0) {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(record.start));
    asm volatile("mov.u32 %0, %%smid;" : "=r"(record.sm));
  }
#endif
  extern __shared__ __align__(1024) unsigned char arena[];
  __shared__ __align__(16) uint64_t raw[SLOTS];
  auto* bars=reinterpret_cast<Barrier*>(raw);
  constexpr Ring qr{Q_VER,Q_READY,-1,Q_OFFSET,Q_ISSUER}, kr{K_VER,K_READY,K_FREE,K_OFFSET,K_ISSUER},
                 vr{V_VER,V_READY,V_FREE,V_OFFSET,V_ISSUER};
  if (threadIdx.x==0) {
    tl::prefetch_tma_descriptor(Q); tl::prefetch_tma_descriptor(K); tl::prefetch_tma_descriptor(V);
    #pragma unroll
    for (int i=0;i<SLOTS;++i) bars[i].init(ARRIVALS[i]);
  }
  tl::fence_barrier_init();
  __syncthreads();
  const int remainder=nk%128;
  const bool short_kv=SHORT_KV && remainder>0 && remainder<=8;
  const int bulk_tiles=tiles-int(short_kv);
  const int diagonal_tile=int(fa3_query_block())/(128/QUERY_ROWS);
  const int active_tiles=CAUSAL ? min(diagonal_tile+1,bulk_tiles) : bulk_tiles;
  int tid=int(threadIdx.x)-C_BASE;
  if (tid>=0 && tid<C_THREADS) {
    tl::warpgroup_reg_alloc<C_REGS>();
    qr.load<QUERY_ROWS,1>(Q,0,fa3_query_block()*QUERY_ROWS,tid,arena,bars);
    float o[64]={}, s[64], m[2]={-CUDART_INF_F,-CUDART_INF_F}, l[2]={};
    half_t p[64];
    auto* q=reinterpret_cast<half_t*>(arena+Q_OFFSET);
    auto scores=[&](int t) {
#include "native_fa3_scores.inc"
      bars[K_FREE+kr.slot(t)].arrive();
    };
#ifdef FRISK_FA3_DEFER_K
    auto scores_defer_k=[&](int t) {
#include "native_fa3_scores.inc"
    };
#endif
    auto accumulate=[&](int t) {
      bars[V_READY+vr.slot(t)].wait(vr.phase(t));
      pv(o,p,reinterpret_cast<half_t*>(arena+V_OFFSET+vr.slot(t)*TILE_BYTES));
      bars[V_FREE+vr.slot(t)].arrive();
    };
    bars[Q_READY].wait(0);
    if constexpr (SAME_TILE_PV) {
      #pragma unroll 1
      for (int t=0;t<active_tiles;++t) {
        FRISK_FA3_SAME_TILE_EVENTS
      }
    } else {
      scores(0);
      update(s,p,o,m,l);
      #pragma unroll 1
      for (int t=0;t<active_tiles-1;++t) {
        FRISK_FA3_LOOKAHEAD_EVENTS
      }
      accumulate(active_tiles-1);
    }
    if (short_kv) {
      if (remainder==1) extend_keys<1>(o,m,l,tid,fa3_query_block(),nq,nk,shape.q,shape.k,shape.v,shape.qp,shape.kp,shape.vp);
      else extend_keys<8>(o,m,l,tid,fa3_query_block(),nq,nk,shape.q,shape.k,shape.v,shape.qp,shape.kp,shape.vp);
    }
    // O is narrowed to binary16 after normalize. For binary32 inputs, a
    // binary64 reciprocal and product preserve the subsequent RN32 -> RN16
    // result. See the midpoint argument in fa3_query_tile.py.
#if !defined(FRISK_ROW_RECIPROCAL) && defined(FRISK_NARROW_OUTPUT_DIV)
    const double inverse[2]={__drcp_rn(double(l[0])),__drcp_rn(double(l[1]))};
    auto normalize=[&](int i) {
      return __double2float_rn(__dmul_rn(double(o[i]),inverse[(i&3)>>1]));
    };
#elif defined(FRISK_ROW_RECIPROCAL)
    float inverse[2]={__frcp_rn(l[0]),__frcp_rn(l[1])};
    auto normalize=[&](int i) { return __fmul_rn(o[i],inverse[(i&3)>>1]); };
#else
    auto normalize=[&](int i) { return o[i]/l[(i&3)>>1]; };
#endif
#ifdef FRISK_REGISTER_WRITEBACK
    // Each accumulator holder writes its own adjacent pair of FP16 values.
    auto* base=output+fa3_head()*shape.output_pitch*128+fa3_query_block()*(QUERY_ROWS*128);
    #pragma unroll
    for (int i=0;i<64;i+=2) {
      const int row=(tid/32)*16+((tid&31)/4)+((i/2)%2)*8;
      const int col=(i/4)*8+(tid%4)*2;
      if ((!dynamic_geometry && ROWS%QUERY_ROWS==0) || fa3_query_block()*QUERY_ROWS+row<nq)
        *reinterpret_cast<uint32_t*>(base+row*128+col)=__pack_half2(
            half_t(normalize(i)),half_t(normalize(i+1)));
    }
#else
    if constexpr (O_ALIAS_BARRIER >= 0) tl::__sync_thread_partial(O_ALIAS_BARRIER,C_THREADS);
    auto* out=reinterpret_cast<half_t*>(arena+O_OFFSET);
    #pragma unroll
    for (int i=0;i<64;++i) o[i]=normalize(i);
    #pragma unroll
    for (int i=0;i<8;++i) {
      auto* src=o+i*8;
      tl::ptx_stmatrix_m8n8_x4(out+(tid/32)*2048+(tid%16)*128+i*16+((tid%32)/16)*8,
          __pack_half2(half_t(src[0]),half_t(src[1])),__pack_half2(half_t(src[2]),half_t(src[3])),
          __pack_half2(half_t(src[4]),half_t(src[5])),__pack_half2(half_t(src[6]),half_t(src[7])));
    }
    tl::fence_proxy_async();  // Publish every writer to the TMA async proxy.
    tl::__sync_thread_partial(O_PUBLISH_BARRIER,C_THREADS);
    if (tid==O_ISSUER) {
      tl::tma_store(output+fa3_head()*shape.output_pitch*128+fa3_query_block()*(QUERY_ROWS*128),out,Q_TILE_BYTES);
      tl::tma_store_arrive(); tl::tma_store_wait<O_WAIT,bool(O_WAIT_READ)>();
    }
#endif
  } else {
    tl::warpgroup_reg_dealloc<P_REGS>();
    int local=int(threadIdx.x)-P_BASE;
    kr.load(K,0,0,local,arena,bars);
    #pragma unroll 1
    for (int t=0;t<active_tiles-1;++t) {
      kr.load(K,t+1,(t+1)*128,local,arena,bars);
      vr.load(V,t,t*128,local,arena,bars);
    }
    vr.load(V,active_tiles-1,(active_tiles-1)*128,local,arena,bars);
  }
#ifdef FRISK_TRACE_CTAS
  __syncthreads();
  if (threadIdx.x==0) {
    record.iterations=active_tiles;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(record.end));
  }
#endif
}
