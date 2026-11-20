# 编译后存活工作、通信边际费用与 native 调度修复

本阶段只在 H100_direct 的 /home/baopeihua/frisk 修改代码。不提交、不修改之前已封存的两个负面验收结果。默认仍为 legacy；实现完成不等同于性能验收通过。

## 实现边界

静态 v6 使用 Frisk 的 GF(2) 消费需求和查询重数，保留原 emitted 账本，另证明 whole-SSA 的存活闭包。根包括最终 store（LI 则只包括自身 producer roots）以及实际发射的 Shared/Shuffle 通信脚手架。不假设 lane 谓词裁剪、跨 SSA CSE、地址 CSE 或 ptxas 寄存器分配。declared scratch 独立保留。

新模型 sm90-layout-surviving-calibrated-v1 有20个特征：原17列加 Shared warp issues、Shuffle warp issues、通信地址 ALU warp issues。移除旧 load-groups 的计价。通信 issues、屏障必须有正系数；四类转换字节和重复 aggregate bank 列固定0以免重复计价。ALU 的1tick工程先验明确不是硬件周期。

编译反馈层分析最终 cubin，绑定 nvdisasm JSON、cuobjdump、分析器源码和工具身份。native 指令字节/数学/依赖链/实际 load-use 距离只是静态结构特征，不是动态执行次数或 cycles。未知指令、分支/调用及工具缺失有 typed 结果。bank 特征继续使用 Frisk 的 GF(2) 地址证明；native LDS/STS 字节必须分别与实际发射标量传输匹配。证明实现源码也进入 profile contract。

## 训练与开发

全部历史34+36 group-unique精确镜像现均为开发数据；只使用可明确分析的镜像拟合 native，其他结果保留为 unsupported。static v6 重放所有历史纯f32完整 assignment，验证同一执行输出的 TTGIR/config/cubin标签身份，不重新搜索历史候选。

独立通信训练注册16个 PTX 程序：baseline、屏障8/32/128、Shuffle8/32/128，以及 Shared重复8/32/128×stride1/4/8。128线程，位精确 uint32 参考、输入输出 guards、同一 GPU1，10 paired batches×5图重放，每图1024实际节点。Shared 差量按同 batch 扣除对应屏障对照，并联合 bootstrap。正的族级95%下界的10%作为保守通信先验，不宣称普适硬件下界；差量包括维持活跃的 XOR/address/依赖协议。

native 目标使用同程序成对相对差量，各程序组等总权重；加固定0.05的相对绝对量正则项。ridge=1e-6、非负约束。使用纯NumPy既有NNLS扩展，旧默认调用数值语义保留。最初 absolute 拟合失败及后续改动均保留在新 artifacts 的开发记录；此时尚未读取新的性能留出标签。

## 冻结与复验

新六例：diamond256×32、remat128×64、conflict256×32、mixed256×64、div128×64、exp128×32。完整 case/shape/tail/ABI/dtype/threads 及 case/tail/ABI/dtype/threads/N 均与全部历史规格去重。Attention64×16 三 ABI 为已观察回归锚点，不计入主几何均值。CPU capability/coverage 预检允许并保留，不允许在看到留出计时后调权、改阈值或重选留出。

每程序固定 legacy/v5d/surviving × LI/J-new 共6来源。每次求解最多65536 probes、256 refinement；完整 portfolio 总量最多393216 probes、1536 refinement，最多6次compile请求。所有准备、求解、编译cache及native分析成本显式记账。相对于单个 legacy-J-new 的GPU收益不冒称同总搜索成本收益。完全同池 native-LI/native-J机会对照必须同选，不能用其证明J独有优势。

所有54来源通过GPU严格数值/ABI/guard/stream/graph及四sanitizer门槛后，先冻结 native 决策，再计时。任一候选 native/bank/训练零列不支持，整个程序统一回退 surviving-J-new，不删除候选。mixed/div 和三 Attention 在 native 调度域外预先声明回退；diamond/remat/conflict/exp 四例子集仅描述，不替代六例primary。

计时复用既有严格引擎：10随机paired batches、每镜像5event样本、同一输入/参考/偏移、实际图节点核验、外来PID整batch无效且不补测。精确 cubin+entry+config+物理ABI去重，保留策略aliases。

门槛保持：六新自然图的有限候选池 regret GM 上界≤1.03、最差上界≤1.10；primary native-pool/legacy-J-new 几何时延比上界<1/1.03；六新+三锚点九项 familywise(alpha=.05/9) case guard上界≤1.05。参考仅为已编译池，不称全域GPU最优。未全部通过则保留现有默认，不冻结对 Triton/TileLang 的外部公平对比版本。

## 参考边界

NVIDIA 官方 CUDA Binary Utilities 说明 nvdisasm 提供反汇编、控制流及寄存器活跃区间工具：https://docs.nvidia.com/cuda/archive/12.8.2/cuda-binary-utilities/index.html 。本阶段不解码未公开调度位，也不把SASS顺序距离等同硬件延迟。Frisk保留GF(2)消费敏感代数需求、图级联合选择及独立执行证明，并用有限编译反馈校正编译器后端改变；不声称GF(2)、核空间或bank分析为Frisk独有。
