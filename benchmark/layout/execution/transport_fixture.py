def encoding(swap=False,mode="base"):
 replica=mode=="replica"
 basis=([1,2,4,0]+[1<<i for i in range(3,10)]) if replica else [1<<i for i in range(10)]
 if swap:
  j=4 if replica else 3;k=9 if replica else 8;basis[j],basis[k]=basis[k],basis[j]
 matrix=[]
 for ob in range(10):
  row=[]
  for b in basis:
   encoded=b if mode=="unit" else (b>>4)|((b&15)<<6);row.append('true' if encoded>>ob&1 else 'false')
  matrix.append('['+','.join(row)+']')
 result='#frisk.distributed<map=#frisk.bit_linear<inputs=["register","lane","warp"],input_bits=[3,5,2],outputs=["dim0","dim1"],output_bits=[6,4],matrix=dense<['+','.join(matrix)+']> : tensor<10x10xi1>>,topology=[8,32,4,1,1],replication=1>'
 if replica:result=result.replace('input_bits=[3,5,2]','input_bits=[4,5,2]').replace('10x10xi1','10x11xi1').replace('topology=[8,','topology=[16,').replace('replication=1','replication=2')
 if mode=='unit':result=result.replace('output_bits=[6,4]','output_bits=[0,10]')
 return result
def fixture(swap=False,cast=False,mode='base'):
 a,b=encoding(swap,mode),encoding(not swap,mode);ty='f16' if cast else 'f32';size=2048 if cast else 4096;width=2 if cast else 4
 header='#A='+a+'\n#B='+b+'\n'
 def storage(name,elementbytes,footprint):return '#'+name+'=#frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[64,16],outputs=["byte_offset","bit_offset"],output_extents=['+str(footprint)+',8],map=affine_map<(i,j)->('+str(elementbytes)+'*(16*i+j),0)>>,memory_space=#frisk<memory_space Global>,alignment=1,vector_granularity=1>\n'
 header+=storage('F',4,4096)+storage('O',width,size)
 body='''module attributes {frisk.target="sm_90a"} {
 "frisk.kernel"() <{sym_name="transport",function_type=(memref<64x16xf32,1>,memref<64x16xf32,1>,memref<64x16xf32,1>,memref<64x16xTY,1>)->()}> ({
 ^bb0(%q:memref<64x16xf32,1>,%k:memref<64x16xf32,1>,%v:memref<64x16xf32,1>,%o:memref<64x16xTY,1>):
 frisk.storage_contract %q {layout=#F}:memref<64x16xf32,1>
 frisk.storage_contract %k {layout=#F}:memref<64x16xf32,1>
 frisk.storage_contract %v {layout=#F}:memref<64x16xf32,1>
 frisk.storage_contract %o {layout=#O}:memref<64x16xTY,1>
 "frisk.parallel"() <{ranges=array<i64:1>,threads=128:i64}> ({
 ^bb0(%i:index):
 %qv=frisk.layout_view %q {layout=#F}:memref<64x16xf32,1> -> memref<64x16xf32,1>
 %ov=frisk.layout_view %o {layout=#O}:memref<64x16xTY,1> -> memref<64x16xTY,1>
 %x=frisk.tile_load %qv : memref<64x16xf32,1> -> tensor<64x16xf32,#A>
 %y=frisk.convert_layout %x : tensor<64x16xf32,#A> -> tensor<64x16xf32,#B>
 CAST
 frisk.tile_store %RESULT,%ov : tensor<64x16xTY,#B>,memref<64x16xTY,1>
 frisk.end
 }) : ()->()
 frisk.end
 }) : ()->()
}
'''.replace('TY',ty).replace('CAST','%z=arith.truncf %y : tensor<64x16xf32,#B> to tensor<64x16xf16,#B>' if cast else '').replace('%RESULT','%z' if cast else '%y')
 result=header+body
 if mode=='unit':result=result.replace('64x16','1x1024').replace('input_extents=[64,16]','input_extents=[1,1024]').replace('16*i+j','1024*i+j')
 return result
