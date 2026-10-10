#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

static void check(cudaError_t error) {
  if (error != cudaSuccess) { std::cerr << cudaGetErrorString(error) << '\n'; std::exit(1); }
}
static void checkDriver(CUresult error) {
  if (error != CUDA_SUCCESS) { const char *text = nullptr; cuGetErrorString(error, &text); std::cerr << (text ? text : "CUDA driver error") << '\n'; std::exit(1); }
}
// Independent coordinate reference: swap coordinate bit 0 with bit 1 (warp)
// or bit 5 (cross-warp). No inferred planner table or generated source is used.
__device__ unsigned coordinate(unsigned t, unsigned bit) {
  unsigned different = ((t >> bit) ^ t) & 1;
  return t ^ different ^ (different << bit);
}
template<class T> __device__ T shuffleBits(T value, unsigned lane) {
  uint64_t bits = static_cast<uint64_t>(value);
  uint32_t low = __shfl_sync(0xffffffff, uint32_t(bits), lane);
  uint32_t high = sizeof(T) == 8 ? __shfl_sync(0xffffffff, uint32_t(bits >> 32), lane) : 0;
  return T(uint64_t(low) | (uint64_t(high) << 32));
}
template<class T, int Mode> __device__ void reference_coordinate(const T *input, T *output, unsigned volume) {
  unsigned t = threadIdx.x;
  if constexpr (Mode == 0) {
    if (t < volume) output[t] = input[t];
  } else if constexpr (Mode == 1) {
    unsigned lane = coordinate(t & 31, 1);
    T value = shuffleBits(input[t % volume], lane);
    if (t < volume) output[lane] = value;
  } else {
    __shared__ T scratch[64];
    scratch[t] = input[t];
    __syncthreads();
    unsigned destination = coordinate(t, 5);
    output[destination] = scratch[destination];
  }
}
#define REFERENCE(NAME, TYPE, MODE, VOLUME) \
extern "C" __global__ void reference_##NAME(const TYPE *input, TYPE *output) { \
  reference_coordinate<TYPE, MODE>(input, output, VOLUME); \
}
REFERENCE(identity_f32, uint32_t, 0, 32)
REFERENCE(shuffle32_f32, uint32_t, 1, 32)
REFERENCE(shared64_f32, uint32_t, 2, 64)
REFERENCE(replicated64_f32, uint32_t, 1, 32)
REFERENCE(shuffle32_i8, uint8_t, 1, 32)
REFERENCE(shuffle32_i16, uint16_t, 1, 32)
REFERENCE(shuffle32_i32, uint32_t, 1, 32)
REFERENCE(shuffle32_i64, uint64_t, 1, 32)
REFERENCE(shuffle32_f16, uint16_t, 1, 32)
REFERENCE(shuffle32_bf16, uint16_t, 1, 32)
REFERENCE(shuffle32_f64, uint64_t, 1, 32)
#ifndef FRISK_REFERENCE_ONLY
int main(int argc, char **argv) {
  if (argc != 11) {
    std::cerr << "usage: runtime CUBIN SYMBOL BYTES THREADS VOLUME MODE DEVICE SAMPLES CASE REFERENCE_CUBIN\n";
    return 1;
  }
  int bytes = std::stoi(argv[3]), threads = std::stoi(argv[4]), volume = std::stoi(argv[5]);
  int mode = std::stoi(argv[6]), device = std::stoi(argv[7]), samples = std::stoi(argv[8]);
  if ((bytes != 1 && bytes != 2 && bytes != 4 && bytes != 8) || (threads != 32 && threads != 64) ||
      (volume != 32 && volume != 64) || mode < 0 || mode > 2 || samples < 0) return 1;
  check(cudaSetDevice(device));
  cudaDeviceProp properties;
  check(cudaGetDeviceProperties(&properties, device));
  if (properties.major != 9 || properties.minor != 0) { std::cerr << "SM90 required\n"; return 1; }
  check(cudaFree(nullptr));
  CUmodule module;
  CUfunction inferred;
  checkDriver(cuModuleLoad(&module, argv[1]));
  checkDriver(cuModuleGetFunction(&inferred, module, argv[2]));
  CUmodule referenceModule;
  CUfunction reference;
  checkDriver(cuModuleLoad(&referenceModule, argv[10]));
  checkDriver(cuModuleGetFunction(&reference, referenceModule, (std::string("reference_") + argv[9]).c_str()));
  void *input = nullptr, *output = nullptr;
  size_t size = size_t(bytes) * volume;
  check(cudaMalloc(&input, size)); check(cudaMalloc(&output, size));
  auto launch = [&](bool useInferred) {
    void *arguments[] = {&input, &output};
    checkDriver(cuLaunchKernel(useInferred ? inferred : reference, 1, 1, 1, threads, 1, 1, 0, nullptr, arguments, nullptr));
  };
  std::vector<uint8_t> host(size), observed(size);
  const uint64_t boundary[] = {0, 0x8000000000000000ull, 0x7ff0000000000000ull,
    0xfff0000000000000ull, 0x7ff8000000000042ull, 1, UINT64_MAX,
    0x000000007fc00042ull, 0x0000000080000000ull, 0x000000007f800000ull,
    0x00000000ff800000ull, 0x0000000000007e42ull, 0x0000000000007f81ull,
    0x0000000000008000ull, 0x0000000000007c00ull};
  for (uint64_t seed : {1ull, 7ull, 42ull, 20260816ull}) {
    for (unsigned variant = 0; variant < 2; ++variant) {
      uint64_t state = seed;
      for (int i = 0; i < volume; ++i) {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        uint64_t value = variant ? boundary[i % (sizeof(boundary) / sizeof(boundary[0]))] : state;
        std::memcpy(host.data() + i * bytes, &value, bytes);
      }
      check(cudaMemcpy(input, host.data(), size, cudaMemcpyHostToDevice));
      for (bool useInferred : {false, true}) {
        check(cudaMemset(output, 0xa5, size));
        launch(useInferred);
        check(cudaDeviceSynchronize());
        check(cudaMemcpy(observed.data(), output, size, cudaMemcpyDeviceToHost));
        if (observed != host) {
          auto index = std::mismatch(observed.begin(), observed.end(), host.begin()).first - observed.begin();
          std::cerr << "correctness failure case=" << argv[9] << " inferred=" << useInferred << " seed=" << seed << " variant=" << variant << " byte=" << index << '\n';
          return 1;
        }
      }
    }
  }
  std::vector<double> refSamples, inferredSamples;
  if (samples) {
    for (unsigned i = 0; i < 100; ++i) { launch(false); launch(true); }
    check(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
    for (int i = 0; i < samples; ++i)
      for (unsigned side = 0; side < 2; ++side) {
        bool useInferred = (i + side) % 2;
        check(cudaEventRecord(start)); launch(useInferred); check(cudaEventRecord(stop));
        check(cudaEventSynchronize(stop));
        float milliseconds;
        check(cudaEventElapsedTime(&milliseconds, start, stop));
        (useInferred ? inferredSamples : refSamples).push_back(double(milliseconds) * 1e6);
      }
    check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
  }
  auto print = [](const std::vector<double> &values) {
    std::cout << '[';
    for (size_t i = 0; i < values.size(); ++i) { if (i) std::cout << ','; std::cout << values[i]; }
    std::cout << ']';
  };
  std::cout << "{\"status\":\"success\",\"correctness\":\"pass\",\"reference_samples_ns\":";
  print(refSamples);
  std::cout << ",\"inferred_samples_ns\":"; print(inferredSamples); std::cout << "}\n";
  check(cudaFree(input)); check(cudaFree(output)); checkDriver(cuModuleUnload(module)); checkDriver(cuModuleUnload(referenceModule));
}
#endif
