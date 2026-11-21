// Extend an existing online-softmax accumulator with at most eight logical keys.
// The certified m64n128 holder map assigns each logical row to a four-lane group.
template<int R>
__device__ __forceinline__ void extend_keys(float (&o)[64],float (&m)[2],float (&l)[2],int tid,
    int qb,int nq,int nk,const half_t* q,const half_t* k,const half_t* v,int qp,int kp,int vp) {
  const int begin=nk/128*128, h=fa3_head();
  if (CAUSAL && (qb+1)*128<=begin) return;
  constexpr float scale=0x1.0527dbd5cafffp-3f;
  #pragma unroll
  for (int r=0;r<2;++r) {
    int row=qb*128+(tid/32)*16+(tid%32)/4+r*8;
    // All four lanes holding a row take the same branch. Each participating subgroup
    // has a separate shuffle mask, including on the causal diagonal.
    bool live=row<nq && (!CAUSAL || row>=begin);
    unsigned mask=15u<<((tid%32)&~3);
    if (live) {
      int count=min(nk-begin,CAUSAL?row-begin+1:R);
      float dots[R],next=m[r];
      #pragma unroll
      for (int j=0;j<R;++j) {
        float dot=0;
        if (j<count) {
          #pragma unroll
          for (int t=0;t<32;++t) {
            int col=(t/2)*8+(tid%4)*2+t%2;
            dot=fmaf(float(q[(h*qp+row)*128+col]),float(k[(h*kp+begin+j)*128+col]),dot);
          }
          dot+=__shfl_xor_sync(mask,dot,1,4);
          dot+=__shfl_xor_sync(mask,dot,2,4);
          next=fmaxf(next,dot);
        }
        dots[j]=dot;
      }
      float a=attention_exp2(m[r]*scale-next*scale),den=l[r]*a;
      #pragma unroll
      for (int t=0;t<32;++t) o[(t/2)*4+2*r+t%2]*=a;
      #pragma unroll
      for (int j=0;j<R;++j) if (j<count) {
        float p=attention_exp2(dots[j]*scale-next*scale);
        den+=p;
        #pragma unroll
        for (int t=0;t<32;++t) {
          int col=(t/2)*8+(tid%4)*2+t%2, ix=(t/2)*4+2*r+t%2;
          o[ix]=fmaf(float(half_t(p)),float(v[(h*vp+begin+j)*128+col]),o[ix]);
        }
      }
      m[r]=next;l[r]=den;
    }
  }
}
