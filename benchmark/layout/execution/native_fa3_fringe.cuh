__device__ __forceinline__ float fringe_normalize(float value,float denominator) {
#ifdef FRISK_ROW_RECIPROCAL
  return __fmul_rn(value,__frcp_rn(denominator));
#else
  return value/denominator;
#endif
}
// One CTA per live tail query. All accesses use absolute, top-left positions.
constexpr int fringe_warps = 8;
constexpr int fringe_keys = 128;
__device__ __forceinline__ float warp_sum(float x) {
  #pragma unroll
  for (int d=16;d;d/=2) x+=__shfl_xor_sync(0xffffffff,x,d);
  return x;
}
// Each warp owns one eighth of the keys and all 128 value columns.
// Merge each warp's (max, denominator, unnormalized O) exactly once.
extern "C" __global__ void fa3_query_fringe(const half_t* q,const half_t* k,const half_t* v,
    half_t* out,int nq,int nk,int qp,int kp,int vp,int op) {
  const int row=nq/128*128+blockIdx.x, head=blockIdx.y;
  const int lane=threadIdx.x%32, warp=threadIdx.x/32;
  const int limit=CAUSAL ? min(nk,row+1) : nk;
  const int begin=0,end=limit;
  constexpr float scale=0x1.0527dbd5cafffp-3f;
  __shared__ float weights[fringe_keys],maxima[fringe_warps],sums[fringe_warps],values[fringe_warps][128];
  float qr[4],acc[4]={},m=-CUDART_INF_F,l=0;
  #pragma unroll
  for (int d=0;d<4;++d) qr[d]=float(q[(head*qp+row)*128+d*32+lane]);
  for (int start=begin;start<end;start+=fringe_keys) {
    float tile_max=-CUDART_INF_F;
    for (int i=warp;i<fringe_keys;i+=fringe_warps) {
      int key=start+i;float s=-CUDART_INF_F;
      if (key<end) {
        float dot=0;
        #pragma unroll
        for (int d=0;d<4;++d) dot=fmaf(qr[d],float(k[(head*kp+key)*128+d*32+lane]),dot);
        s=warp_sum(dot)*scale;
      }
      if (lane==0) weights[i]=s;
      tile_max=fmaxf(tile_max,s);
    }
    __syncwarp();
    float next=fmaxf(m,tile_max);
    float alpha=(m==-CUDART_INF_F)?0:attention_exp2(m-next);
    l*=alpha;
    #pragma unroll
    for (int d=0;d<4;++d) acc[d]*=alpha;
    for (int i=warp;i<min(fringe_keys,end-start);i+=fringe_warps) {
      float p=attention_exp2(weights[i]-next);l+=p;
      #pragma unroll
      for (int d=0;d<4;++d)
        acc[d]=fmaf(float(half_t(p)),float(v[(head*vp+start+i)*128+d*32+lane]),acc[d]);
    }
    __syncwarp();m=next;
  }
  if (lane==0) { maxima[warp]=m;sums[warp]=l; }
  #pragma unroll
  for (int d=0;d<4;++d) values[warp][d*32+lane]=acc[d];
  __syncthreads();
  if (threadIdx.x<128) {
    float global_m=-CUDART_INF_F,denom=0,val=0;
    for (int w=0;w<fringe_warps;++w) global_m=fmaxf(global_m,maxima[w]);
    for (int w=0;w<fringe_warps;++w) {
      float a=(sums[w]==0)?0:attention_exp2(maxima[w]-global_m);
      denom=fmaf(sums[w],a,denom);val=fmaf(values[w][threadIdx.x],a,val);
    }
    out[(head*op+row)*128+threadIdx.x]=half_t(fringe_normalize(val,denom));
  }
}
extern "C" __global__ void fa3_merge_fringe(const float* partial,half_t* out,int nq,int op,int parts) {
  const int head=blockIdx.y,r=blockIdx.x;
  auto base=partial+(size_t(head)*gridDim.x+r)*parts*130;
  float m=-CUDART_INF_F;
  for (int p=0;p<parts;++p) m=fmaxf(m,base[p*130+128]);
  float d=0,x=0;
  for (int p=0;p<parts;++p) {
    auto s=base+p*130;float a=s[129]==0?0:attention_exp2(s[128]-m);
    d=fmaf(s[129],a,d);x=fmaf(s[threadIdx.x],a,x);
  }
  out[(head*op+nq/128*128+r)*128+threadIdx.x]=half_t(fringe_normalize(x,d));
}

// A split covers at most 64 keys; eight warps cooperate on one query.
extern "C" __global__ void fa3_fringe_partial(const half_t* q,const half_t* k,const half_t* v,
    int nq,int nk,int qp,int kp,int vp,float* partial) {
  const int head=blockIdx.y,row=nq/128*128+blockIdx.x,warp=threadIdx.x/32,lane=threadIdx.x%32;
  const int limit=CAUSAL?min(row+1,nk):nk, base=blockIdx.z*64+warp*8;
  float qr[4],dots[8],acc[4]={},m=-CUDART_INF_F,l=0;
  #pragma unroll
  for (int j=0;j<4;++j) qr[j]=float(q[(head*qp+row)*128+lane+32*j]);
  #pragma unroll
  for (int i=0;i<8;++i) {
    float dot=0;
    if (base+i<limit) {
      #pragma unroll
      for (int j=0;j<4;++j) dot=fmaf(qr[j],float(k[(head*kp+base+i)*128+lane+j*32]),dot);
      dot=warp_sum(dot)*0x1.0527dbd5cafffp-3f;
    } else dot=-CUDART_INF_F;
    dots[i]=dot;m=fmaxf(m,dot);
  }
  #pragma unroll
  for (int i=0;i<8;++i) if (base+i<limit) {
    float p=attention_exp2(dots[i]-m); l+=p;
    #pragma unroll
    for (int j=0;j<4;++j) acc[j]=fmaf(float(half_t(p)),float(v[(head*vp+base+i)*128+lane+32*j]),acc[j]);
  }
  __shared__ float mx[8],den[8],val[8][128];
  if (lane==0) {mx[warp]=m;den[warp]=l;}
  #pragma unroll
  for (int j=0;j<4;++j) val[warp][lane+j*32]=acc[j];
  __syncthreads();
  if (threadIdx.x>=128) return;
  float top=-CUDART_INF_F,total=0,x=0;
  #pragma unroll
  for (int w=0;w<8;++w) top=fmaxf(top,mx[w]);
  #pragma unroll
  for (int w=0;w<8;++w) {
    float a=den[w]==0?0:attention_exp2(mx[w]-top);
    total=fmaf(den[w],a,total); x=fmaf(val[w][threadIdx.x],a,x);
  }
  auto dst=partial+((size_t(head)*gridDim.x+blockIdx.x)*gridDim.z+blockIdx.z)*130;
  dst[threadIdx.x]=x;
  if (!threadIdx.x) {dst[128]=top;dst[129]=total;}
}
