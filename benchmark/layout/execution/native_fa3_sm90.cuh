// SM90 FA3 replay. Constants in the companion translation unit come from
// the Frisk execution contract and the two verified MMA descriptor plans.
#include <tl_templates/cuda/instruction/wgmma.h>
#include <tl_templates/cuda/intrin.h>
#include <tl_templates/cuda/barrier.h>
#include <tl_templates/cuda/copy_sm90.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <math_constants.h>

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
  int versions, ready, free_slot, byte_offset;
  __device__ int slot(int i) const { return i % versions; }
  __device__ int phase(int i) const { return (i / versions) & 1; }
  __device__ void load(const CUtensorMap& map, int tile, int row, int local,
                       unsigned char* arena, Barrier* bars) const {
    if (free_slot >= 0 && tile >= versions)
      bars[free_slot+slot(tile)].wait(phase(tile-versions));
    if (local == 0) {
      auto& b=bars[ready+slot(tile)];
      b.arrive_and_expect_tx(TILE_BYTES);
      auto* dst=reinterpret_cast<half_t*>(arena+byte_offset+slot(tile)*TILE_BYTES);
      tl::tma_load(map,b,dst,0,row,int(fa3_head()),0);
      tl::tma_load(map,b,dst+8192,64,row,int(fa3_head()),0);
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
  constexpr Ring qr{1,Q_READY,-1,Q_OFFSET}, kr{K_VER,K_READY,K_FREE,K_OFFSET},
                 vr{V_VER,V_READY,V_FREE,V_OFFSET};
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
  const int active_tiles=CAUSAL ? min(int(fa3_query_block())+1,bulk_tiles) : bulk_tiles;
  int tid=int(threadIdx.x)-C_BASE;
  if (tid>=0 && tid<C_THREADS) {
    tl::warpgroup_reg_alloc<C_REGS>();
    qr.load(Q,0,fa3_query_block()*128,tid,arena,bars);
    float o[64]={}, s[64], m[2]={-CUDART_INF_F,-CUDART_INF_F}, l[2]={};
    half_t p[64];
    auto* q=reinterpret_cast<half_t*>(arena+Q_OFFSET);
    auto scores=[&](int t) {
      bars[K_READY+kr.slot(t)].wait(kr.phase(t));
      qk(s,q,reinterpret_cast<half_t*>(arena+K_OFFSET+kr.slot(t)*TILE_BYTES),tid);
      if constexpr (CAUSAL) {
        if (t==int(fa3_query_block())) {
          #pragma unroll
          for (int i=0;i<64;++i) {
            int row=(tid/32)*16+((tid&31)/4)+((i/2)%2)*8;
            int col=(i/4)*8+(tid%4)*2+i%2;
            if (col>row) s[i]=-CUDART_INF_F;
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
      bars[K_FREE+kr.slot(t)].arrive();
    };
    auto accumulate=[&](int t) {
      bars[V_READY+vr.slot(t)].wait(vr.phase(t));
      pv(o,p,reinterpret_cast<half_t*>(arena+V_OFFSET+vr.slot(t)*TILE_BYTES));
      bars[V_FREE+vr.slot(t)].arrive();
    };
    bars[Q_READY].wait(0);
    if constexpr (SAME_TILE_PV) {
      #pragma unroll 1
      for (int t=0;t<active_tiles;++t) {
        scores(t);
        update(s,p,o,m,l);
        accumulate(t);
      }
    } else {
      scores(0);
      update(s,p,o,m,l);
      #pragma unroll 1
      for (int t=0;t<active_tiles-1;++t) {
        scores(t+1);
        accumulate(t);
        update(s,p,o,m,l);
      }
      accumulate(active_tiles-1);
    }
    if (short_kv) {
      if (remainder==1) extend_keys<1>(o,m,l,tid,fa3_query_block(),nq,nk,shape.q,shape.k,shape.v,shape.qp,shape.kp,shape.vp);
      else extend_keys<8>(o,m,l,tid,fa3_query_block(),nq,nk,shape.q,shape.k,shape.v,shape.qp,shape.kp,shape.vp);
    }
    // The optional numerical contract rounds one reciprocal per row.
#ifdef FRISK_ROW_RECIPROCAL
    float inverse[2]={__frcp_rn(l[0]),__frcp_rn(l[1])};
    auto normalize=[&](int i) { return __fmul_rn(o[i],inverse[(i&3)>>1]); };
#else
    auto normalize=[&](int i) { return o[i]/l[(i&3)>>1]; };
#endif
#ifdef FRISK_REGISTER_WRITEBACK
    // Each accumulator holder writes its own adjacent pair of FP16 values.
    auto* base=output+fa3_head()*shape.output_pitch*128+fa3_query_block()*16384;
    #pragma unroll
    for (int i=0;i<64;i+=2) {
      const int row=(tid/32)*16+((tid&31)/4)+((i/2)%2)*8;
      const int col=(i/4)*8+(tid%4)*2;
      if ((!dynamic_geometry && ROWS%128==0) || fa3_query_block()*128+row<nq)
        *reinterpret_cast<uint32_t*>(base+row*128+col)=__pack_half2(
            half_t(normalize(i)),half_t(normalize(i+1)));
    }
#else
    if constexpr (O_OFFSET == Q_OFFSET) tl::__sync_thread_partial(2,C_THREADS);
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
    tl::__sync_thread_partial(3,C_THREADS);
    if (tid==0) {
      tl::tma_store(output+fa3_head()*shape.output_pitch*128+fa3_query_block()*16384,out,TILE_BYTES);
      tl::tma_store_arrive(); tl::tma_store_wait<0,true>();
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
