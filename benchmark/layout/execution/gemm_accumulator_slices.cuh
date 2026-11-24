#include <cuda_runtime.h>
#include <stdint.h>

// The native launch contract fixes the CTA at two 128-thread warpgroups.
constexpr unsigned kCopyThreads = 256;

template<bool CacheAll>
__device__ __forceinline__ void copy16_async(unsigned char* dst, const uint16_t* src) {
  unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  if constexpr (CacheAll) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(shared), "l"(src) : "memory");
  } else {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(shared), "l"(src) : "memory");
  }
}

// One vector is eight adjacent f16 values. Coordinates and descriptor bits
// refer to the complete, original K16 operand maps even for a partial copy.
template<int M, int K, int Begin, int End>
__device__ __forceinline__ void copy_a(unsigned char* sm, const uint16_t* a, int k) {
  constexpr unsigned count = (End-Begin)*2;
  #pragma unroll
  for (unsigned start = 0; start < count; start += kCopyThreads) {
    unsigned v = start + threadIdx.x;
    if (count % kCopyThreads != 0 && v >= count) continue;
    unsigned i = Begin + v/2, j = (v%2)*8;
    unsigned off = i*16 + (j/8)*M*16;
    copy16_async<true>(sm + off, a + i*K + k + j);
  }
}

template<int N, int Begin, int End>
__device__ __forceinline__ void copy_b(unsigned char* sm, const uint16_t* b, int k) {
  constexpr int vectors = (End-Begin)/8;
  constexpr unsigned count = 16*vectors;
  #pragma unroll
  for (unsigned start = 0; start < count; start += kCopyThreads) {
    unsigned v = start + threadIdx.x;
    if (count % kCopyThreads != 0 && v >= count) continue;
    unsigned i = v/vectors, j = Begin + v%vectors*8;
    unsigned off = j%16*2 + j/16*256 + i%8*32 + i/8*N*16;
    copy16_async<false>(sm + (off ^ (((off >> 7) & 1) << 4)), b + (k+i)*N + j);
  }
}

__device__ __forceinline__ uint64_t descriptor(const unsigned char* base, uint64_t bits) {
  return (__cvta_generic_to_shared(base)>>4) | bits;
}

template<int N, int Atom, bool Reduce, bool StripeC, int Axis, int PanelRows, int Row0, int Col0, int Row1, int Col1, int M>
__device__ __forceinline__ void deliver_atom(const float* d, float* c, int row, int col,
                                             float* scratch, float* partials) {
  int tid = threadIdx.x, lane = tid%32, warp = tid/32;
  int r = row + (warp%4)*16 + lane/4;
  // Consecutive registers 2j and 2j+1 own adjacent columns of the same row.
  if constexpr (StripeC) {
    // Drain one source warp at a time. Every destination word is stored
    // by thread (global flat index % 256), including atoms narrower than a CTA.
    for (int group = 0; group < 2; ++group) {
      const int rb = group ? Row1 : Row0, cb = group ? Col1 : Col0;
      #pragma unroll 1
      for (int w = 0; w < 4; ++w) {
        #pragma unroll
        for (int part = 0; part < 16; part += PanelRows) {
          if (warp == group*4+w) {
            #pragma unroll
            for (int j = 0; j < Atom/2; ++j) {
              int y = lane/4 + ((j/2)%2)*8;
              if (y >= part && y < part + PanelRows)
                scratch[(y-part)*Atom + (j/4)*8 + lane%4*2 + j%2] = d[j];
            }
          }
          __syncthreads();
          #pragma unroll
          for (int y = 0; y < PanelRows; ++y) {
            int start = (rb + w*16 + part + y)*N + cb;
            int x = (tid - start) & 255;
            if (x < Atom) c[start+x] = scratch[y*Atom+x];
          }
          __syncthreads();
        }
      }
    }
  } else if ((reinterpret_cast<uintptr_t>(c) & 7) == 0) {
    #pragma unroll
    for (int j = 0; j < Atom/2; j += 2) {
      float* dst = c + (r + (j/2%2)*8)*N + col + (j/4)*8 + (lane%4)*2;
      *reinterpret_cast<float2*>(dst) = make_float2(d[j], d[j+1]);
    }
  } else {
    #pragma unroll
    for (int j=0; j<Atom/2; ++j)
      c[(r+(j/2%2)*8)*N + col + (j/4)*8 + (lane%4)*2 + j%2] = d[j];
  }
  if constexpr (Reduce && Axis == 1) {
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
      float x[Atom/8];
      #pragma unroll
      for (int j = 0; j < Atom/8; ++j) x[j] = __fadd_rn(d[4*j+2*half], d[4*j+2*half+1]);
      #pragma unroll
      for (int step = 1; step < Atom/8; step *= 2) {
        #pragma unroll
        for (int j = 0; j < Atom/8; j += 2*step) x[j] = __fadd_rn(x[j], x[j+step]);
      }
      float v = x[0];
      v = __fadd_rn(v, __shfl_xor_sync(0xffffffff, v, 1));
      v = __fadd_rn(v, __shfl_xor_sync(0xffffffff, v, 2));
      if (!(lane & 3)) partials[(col/Atom)*M+r+8*half] = v;
    }
    __syncthreads();
  } else if constexpr (Reduce) {
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
