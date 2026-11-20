# 支持域与当前 GPU 闭环：整合执行计划

日期：2026-11-02（Asia/Shanghai）。仅在 H100_direct:/home/baopeihua/frisk 修改和保存。保持原 Git HEAD 9f79155，未经新授权不提交/推送。旧封存产物与报告不回写。

## 范围与完成定义

本轮将当前 GF(2) 与成本语义版本的正常推断输出，连续验证为 selected IR → checked execution export → 实际backend支持判定/TTGIR → native cubin → driver加载 → GPU数值/安全/资源证据。声明三个不同层次：通用布局推断/证明域、现有GPU后端能力谓词、已逐例验证的benchmark子集；不将22个实例的通过当成任意形状保证。

主矩阵预登记11个自然程序，各用legacy-v2和显式冻结profile的calibrated-v3，共22配置：diamond32×32、remat128×16、mixed128×32、conflict256×8；diamond32×16、remat32×16、mixed32×16、conflict64×8；固定64×16 Attention的weak-vector、strong-cpasync、strong-TMA。机制图不调用A/B候选注入；Attention三种ABI分层，沿用原硬契约，不跨ABI宣称收益。每例默认candidate生成/传播/搜索及转换优化，保持原预算；所有22例都要求成功，不得临时改shape、手工改布局或删除失败行。

显式profile使用旧冻结文件 cost_resource_calibration_artifacts/run-calibration-r3-20261031T105856Z/fit/profile.json，raw SHA256 6b4947af82c2748fc8db75268ddb621fc45852d238b63deee12cf17c6cde7388；格式v2和当前语义v3分开记录，未重新拟合。

当前执行域主要是SM90a、单CTA/128线程、f16/f32、四Global根Q/K/V/O、受检标量PTX发射、受支持MMA/Reduce/Broadcast/转换及Copy合同。逻辑footprint、导出枚举、arbitrary GF2地址可表达性等限制如实保留；本轮不实现大算子多CTA分块、任意动态shape或流水重叠。

## 一体化实施

1. 建立版本化支持域判定和分阶段拒绝信息，复用真实emitter/证明逻辑，不靠实例白名单放行。未证明/unsupported/malformed分别记录；保留输入和发布原子性。
2. 复用当前backend、native compiler、loaded-driver resource与安全基础，提供本阶段统一可重放driver及独立reference，新增正反例回归。
3. CPU预检完成后冻结source/tool/input/protocol/GPU UUID，再串行在GPU1运行22配置；普通流/非默认流/CUDA graph及适用四sanitizer均针对确切cubin。负例不进入成功数值计数。
4. 完成当前源码回归、实际支持矩阵、独立整体复审、可复现补丁与证据索引。更新长期文档只添加前缀，保留旧文。

## 验证协议

运行ABI要求输出物理区域与所有只读输入区域不重叠；Q/K/V之间可只读重叠。所有22原配置本来使用独立buffer；新增runtime extent/overlap负例明确守门，不将per-thread先读后写当作in-place跨线程安全证明。

- 机制图每个SSA算术分别FP32舍入，mul/sub用bitexact；含exp/div的mixed沿用abs<=1e-5+2e-6*abs(ref)，不得观测失败后放宽。输入至少既有3随机seed及zero/constant/signed-coordinate/cancellation结构样本；除法分母保持非零，有限、无溢出。
- Attention沿用15输入(seed0–9及5结构样本)、FP16输入、FP32逐K累加、稳定Softmax、P的RNE f16转换、FP32 PV，O有限且abs<=0.002+0.002*abs(ref)。
- weak地址offset1/2/3；strong QKV offset0/128/256与O offset1/2/3独立组合。输出sentinel、前后保护区、输入字节不变、重复执行、非默认stream与graph捕获检查。
- 记录compiler/driver寄存器、user Shared、raw/reserved Shared、local/stack/spill的真实语义，实际loaded cubin SHA/entry必须匹配；无编译物理寄存器预测仍not-qualified。
- memcheck/racecheck/synccheck/initcheck，工具能力边界如实记录。安全复用只允许同cubin bytes、完整execution config、physical ABI和相同安全输入协议；每个配置普通数值验证仍执行。
- 不做正式GPU性能比较；编译/CPU wall仅诊断，不把缓存命中称作cold compilation。GPU UUID固定GPU-028d5ad6-2ff8-f6e3-130f-9eefb0447618；不停止外部任务，不使用GPU0。
- 每阶段前后核对source、工具、输入/profile与动态依赖身份。源码改变后开启新运行，不覆盖旧尝试。保存错误、超时和不支持原因。
