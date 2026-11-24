#include <cuda_runtime.h>
#include <stdint.h>

// One vector is eight adjacent f16 values. Coordinates and descriptor bits
// refer to the complete, original K16 operand maps even for a partial copy.
template<int M, int K, int Begin, int End>
__device__ __forceinline__ void copy_a(unsigned char* sm, const uint16_t* a, int k) {
  for (int v = threadIdx.x; v < (End-Begin)*2; v += blockDim.x) {
    int i = Begin + v/2, j = (v%2)*8;
    int off = i*16 + (j/8)*M*16;
    *reinterpret_cast<uint4*>(sm+off) = *reinterpret_cast<const uint4*>(a+i*K+k+j);
  }
}

template<int N, int Begin, int End>
__device__ __forceinline__ void copy_b(unsigned char* sm, const uint16_t* b, int k) {
  constexpr int vectors = (End-Begin)/8;
  for (int v = threadIdx.x; v < 16*vectors; v += blockDim.x) {
    int i = v/vectors, j = Begin + v%vectors*8;
    int off = j%16*2 + j/16*256 + i%8*32 + i/8*N*16;
    *reinterpret_cast<uint4*>(sm+(off^(((off>>7)&1)<<4))) =
        *reinterpret_cast<const uint4*>(b+(k+i)*N+j);
  }
}

__device__ __forceinline__ uint64_t descriptor(const unsigned char* base, uint64_t bits) {
  return (__cvta_generic_to_shared(base)>>4) | bits;
}

template<int N, int Atom, bool Reduce>
__device__ __forceinline__ void deliver_atom(const float* d, float* c, int row, int col,
                                             float* scratch, float* partials) {
  int tid = threadIdx.x, lane = tid%32, warp = tid/32;
  int r = row + (warp%4)*16 + lane/4;
  #pragma unroll
  for (int j=0; j<Atom/2; ++j)
    c[(r+(j/2%2)*8)*N + col + (j/4)*8 + (lane%4)*2 + j%2] = d[j];
  if constexpr (Reduce) {
    // Match the tree in the common replay: row bits 3,0,1,2,4,5,
    // followed by the row-repeat bits. Never reload the external C output.
    #pragma unroll
    for (int j=0; j<Atom/4; ++j) {
      int reg = (j/2)*4+j%2, local_col = (j/2)*8 + (lane%4)*2 + j%2;
      float x = __fadd_rn(d[reg],d[reg+2]);
      x = __fadd_rn(x,__shfl_xor_sync(0xffffffff,x,4));
      x = __fadd_rn(x,__shfl_xor_sync(0xffffffff,x,8));
      x = __fadd_rn(x,__shfl_xor_sync(0xffffffff,x,16));
      if (lane<4) scratch[warp*Atom+local_col] = x;
    }
    __syncthreads();
    const float* w = scratch+(tid/128)*4*Atom;
    for (int j=tid%128; j<Atom; j+=128)
      partials[(row/64)*N+col+j] = __fadd_rn(__fadd_rn(w[j],w[Atom+j]),
                                              __fadd_rn(w[2*Atom+j],w[3*Atom+j]));
    __syncthreads();
  }
}

template<int Rows, int N>
__device__ __forceinline__ float finish_column(const float* partials, int col) {
  float levels[Rows];
  #pragma unroll
  for (int i=0; i<Rows; ++i) levels[i] = partials[i*N+col];
  #pragma unroll
  for (int step=1; step<Rows; step*=2) {
    #pragma unroll
    for (int i=0; i<Rows; i+=2*step) levels[i]=__fadd_rn(levels[i],levels[i+step]);
  }
  return levels[0];
}
