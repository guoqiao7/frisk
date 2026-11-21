// Packed rows with independent logical sequence lengths and physical head pitches.
extern "C" int frisk_launch_workspace(const void* q,const void* k,const void* v,void* o,void* stream,
    int nq,int nk,int heads,int qp,int kp,int vp,int op,CtaTrace* trace,float* workspace,size_t floats) {
  if ((!dynamic_geometry && (nq!=ROWS || nk!=KV_ROWS || heads!=HEADS)) ||
      nq<=128 || nk<=128 || heads<=0 || heads>65535 || qp<nq || op<nq || kp<nk || vp<nk)
    return int(cudaErrorInvalidValue);
#ifdef FRISK_TRACE_CTAS
  if (!trace) return int(cudaErrorInvalidValue);
#endif
  const int pitches[]={qp,kp,vp,op};
  for (int pitch : pitches)
    if (uint64_t(pitch)*heads*128 >= (1ull<<31)) return int(cudaErrorInvalidValue);
  const int parts=(nk+63)/64;
  if (QUERY_FRINGE==3 && nq%128 && (parts>65535 || !workspace ||
      floats<size_t(nq%128)*heads*parts*130)) return int(cudaErrorInvalidValue);
  CUtensorMap maps[3];
  const void* inputs[]={q,k,v};
  cuuint32_t box[]={64,128,1,1}, unit[]={1,1,1,1};
  for (int i=0;i<3;++i) {
    cuuint64_t dims[]={128,cuuint64_t(i==0?nq:nk),cuuint64_t(heads),1};
    cuuint64_t pitch=cuuint64_t(pitches[i])*256;
    cuuint64_t strides[]={256,pitch,pitch*heads};
    auto status=cuTensorMapEncodeTiled(&maps[i],CU_TENSOR_MAP_DATA_TYPE_FLOAT16,4,
          const_cast<void*>(inputs[i]),dims,strides,box,unit,CU_TENSOR_MAP_INTERLEAVE_NONE,
          CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status!=CUDA_SUCCESS) return 10000+int(status);
  }
  auto status=cudaFuncSetAttribute(main_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,ARENA);
  if (status!=cudaSuccess) return int(status);
  const int qblocks=QUERY_FRINGE ? nq/128 : (nq+127)/128;
  main_kernel<<<dim3(qblocks,heads),CTA,ARENA,reinterpret_cast<cudaStream_t>(stream)>>>(
        maps[1],reinterpret_cast<half_t*>(o),maps[0],maps[2],{nq,nk,op,qp,kp,vp,static_cast<const half_t*>(q),static_cast<const half_t*>(k),static_cast<const half_t*>(v)},trace);
  status=cudaGetLastError();
  if (status!=cudaSuccess) return int(status);
  if (QUERY_FRINGE==1 && nq%128) {
    fa3_query_fringe<<<dim3(nq%128,heads),fringe_warps*32,0,reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<const half_t*>(q),static_cast<const half_t*>(k),static_cast<const half_t*>(v),
        static_cast<half_t*>(o),nq,nk,qp,kp,vp,op);
    return int(cudaGetLastError());
  }
  if (QUERY_FRINGE==3 && nq%128) {
    auto st=reinterpret_cast<cudaStream_t>(stream);
    fa3_fringe_partial<<<dim3(nq%128,heads,parts),256,0,st>>>(static_cast<const half_t*>(q),
        static_cast<const half_t*>(k),static_cast<const half_t*>(v),nq,nk,qp,kp,vp,workspace);
    status=cudaGetLastError();
    if (status!=cudaSuccess) return int(status);
    fa3_merge_fringe<<<dim3(nq%128,heads),128,0,st>>>(workspace,static_cast<half_t*>(o),nq,op,parts);
    return int(cudaGetLastError());
  }
  return int(cudaSuccess);
}
extern "C" int frisk_launch_strided(const void* q,const void* k,const void* v,void* o,void* stream,
    int nq,int nk,int heads,int qp,int kp,int vp,int op,CtaTrace* trace) {
  return frisk_launch_workspace(q,k,v,o,stream,nq,nk,heads,qp,kp,vp,op,trace,nullptr,0);
}
extern "C" int frisk_launch(const void* q,const void* k,const void* v,void* o,void* stream) {
  return frisk_launch_strided(q,k,v,o,stream,ROWS,KV_ROWS,HEADS,ROWS,KV_ROWS,KV_ROWS,ROWS,nullptr);
}
