// Driver ABI for packed BHSD FP16 and a full, aligned 128 x 128 tile.
extern "C" int frisk_launch(const void* q,const void* k,const void* v,void* o,void* stream) {
  CUtensorMap maps[3];
  const void* inputs[3]={q,k,v};
  cuuint64_t dims[]={128,8192,16,1}, strides[]={256,2097152,33554432};
  cuuint32_t box[]={64,128,1,1}, unit[]={1,1,1,1};
  for (int i=0;i<3;++i) {
    auto status=cuTensorMapEncodeTiled(&maps[i],CU_TENSOR_MAP_DATA_TYPE_FLOAT16,4,
          const_cast<void*>(inputs[i]),dims,strides,box,unit,CU_TENSOR_MAP_INTERLEAVE_NONE,
          CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status!=CUDA_SUCCESS) return 10000+int(status);
  }
  auto status=cudaFuncSetAttribute(main_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,ARENA);
  if (status!=cudaSuccess) return int(status);
  main_kernel<<<dim3(64,16),CTA,ARENA,reinterpret_cast<cudaStream_t>(stream)>>>(
        maps[1],reinterpret_cast<half_t*>(o),maps[0],maps[2]);
  return int(cudaGetLastError());
}
