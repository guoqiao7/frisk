# 编译后存活工作、通信边际费用与 native 调度：实现及新留出验收报告

**实现和新留出复验已完成，整体选择质量未通过，默认保持 legacy。** 六来源候选池方案相对单次 legacy-J-new 的 kernel 配对几何时间比为 **0.900428329 [0.896961307, 0.903004470]**，约 **1.1106×** 加速，九项5%回归护栏通过；但候选池 regret 几何均值 **1.089051266 [1.084817586, 1.091463306]**、最差 **1.668360678 [1.629822470, 1.690654395]** 超过1.03/1.10门槛。唯一 regret 失败为 conflict256×32 的整池回退。source snapshot 是可重放实验版本，本轮不进入对 Triton/TileLang 的外部公平对比冻结，也没有提交代码。

上述GPU收益采用六次来源求解和至多六次编译请求，额外预算与开销完整披露；不能把它称为相对单次基线的同总预算收益或 J 独有优势。所有新性能留出时延是在源代码、模型、候选、回退和阈值冻结后获得，未进行留出后调权或删例。

代码与证据均位于 H100_direct 的 `/home/baopeihua/frisk`，没有本地代码副本。本轮没有提交代码。上一轮负面结果及两个已封存阶段保留，新的实现和证据写入 `guoqiao/optimize_plan/native_cost_repair_artifacts`。

## 1. 本轮处理的问题与实现边界

上一轮失败包含三个不同层面：发射账本把被 ptxas 删除的未使用 V 加载计入成本；Shared/屏障/地址工作存在零系数而缺乏独立通信增量约束；PTX 发射层 load groups 不能刻画优化后的 native 排序和依赖。扩大搜索只会更充分地优化失真的目标。本轮将这三个层面分开处理，然后通过一套有上限、可审计的公共候选池连接起来。

| 层面 | 本轮实现 | 证据的准确含义 |
|---|---|---|
| 编译前搜索 | GF(2) 消费者需求与 whole-SSA 存活闭包，新增静态 v6 模型 | 可证明剔除部分整 SSA 死值；不声称等同 ptxas 的全部优化 |
| 通信成本 | 实际 Shared/Shuffle warp issues、通信 ALU、方向化 bank burden；独立通信协议增量校准 | 非零经验费用与工程先验；不是通用硬件延时下界 |
| 编译后反馈 | 对已经编译的 cubin 提取存活指令、读写宽度、物理依赖和顺序统计 | 精确绑定的静态 native 证据；不是动态执行次数、cycles 或测得 ILP |
| 选择与公平性 | 至多六个完整赋值共同编译反馈、统一评分、整池回退 | 所测候选池内的选择；不代表原始候选全域最优，也不单独证明 J 优于 LI |

`legacy` 默认保持原值。模型、支持域、预算、回退、阈值均须在新性能标签读取之前冻结；失败或不确定结果也不能用于本轮重调模型后宣称留出通过。

## 2. 静态 v6：保留原发射账本，另建存活证明

`analyzeLayoutSurvivingDemand` 首先构建原有完整发射需求账本，再从最终输出或 LI 的局部根向生产者做闭包。已经实际发射的 Shared/Shuffle 边界被额外纳入根集合，沿转换端点、纯算术操作数和加载继续追溯。闭包之外的整 SSA 值及其读取费用可删除，闭包内部的仿射查询重数保留。输出同时包含 emitted、surviving、保留/删除变量、保留通信边界及删除查询向量数，`native_exact=false`。

该实现是保守的整值存活分析：不实现 query 级 ptxas DCE、CSE、全部常量折叠或 native 重调度；不把未使用的通信骨架及屏障自行删除。Shared 声明 scratch 仍按实际串行分配规则保留，不能把声明资源误写成活跃值峰值。原来的 GF(2) 查询仍为 `q(t)=c xor Σ(t_i a_i)`，保留 SSA 与完整查询键、规范同线程原像和 first-owner 语义。

v6 不继续对过滤后的账本使用 `global_load_issue_groups`。存活 JSON 将该项标记为不可用，native 调度交给后续编译反馈。原 v5d 的18项发射对账保持兼容，静态 v6 使用下列20项计价键：

| 类别 | 字段 |
|---|---|
| 访问 | `access_warp_issues`, `global_sectors32`, `shared_bank_excess` |
| 旧指令通道 | `copy_setup_units`, `mma_instructions`, `reduction_element_ops` |
| 算术 | `alu_warp_issues`, `div_warp_issues`, `exp_warp_issues`, `cast_warp_issues` |
| 转换兼容字段 | `conversion_register_bytes`, `conversion_shuffle_bytes`, `conversion_shared_bytes`, `conversion_conservative_bytes` |
| 同步及 bank | `synchronization_barriers`, `shared_read_bank_excess`, `shared_write_bank_excess` |
| 新增实际费用通道 | `shared_warp_issues`, `shuffle_warp_issues`, `communication_alu_warp_issues` |

当前 profile 将总 `shared_bank_excess` 与四个旧转换字节项固定为0，由读/写 bank 和实际通信 issues 计价，避免重复特征叠加。字段为0不表示相应物理工作免费；通信已经转入有正费用约束的明确通道。所有 profile 版本的字段数、解析、ID 和旧语义有独立校验。

新静态模型延续现有精确纯 f32 单 CTA DAG 支持域，不能据此扩大为完整 Attention 成本模型。纯图范围之外继续保留显式 carrier/fallback 说明；MMA、异步流水和一般控制流没有被包装成精确存活计价。

## 3. 编译后15项 native 证据与14项组合评分

`native_cost_analysis.py` 只读取已编译的 `kernel.cubin`，使用 CUDA12.8 `nvdisasm -json` 和 `cuobjdump --dump-resource-usage`，不编译或执行新内核。NVIDIA 提供原生反汇编、函数边界、JSON 及资源信息；本实现据此建立自己的有界特征与拒绝规则。[NVIDIA CUDA Binary Utilities](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-binary-utilities/index.html#nvdisasm)

原始 native 特征固定为15项：总静态指令数；global/shared/local 的 load/store 声明字节宽度（六项）；barrier、shuffle、float arithmetic、special math、integer/logic 次数（五项）；`load_use_exposure_32`、`load_use_exposure_128`；`dependency_depth`。这里的次数均是原生指令静态出现次数，谓词指令计一次，不宣称所有 lane 的谓词都成立；字节也不等同于 memory sectors 或实际流量。

调度 DAG 包含物理寄存器 RAW/WAW/WAR、条件定义对旧值的保留、宽寄存器对/四元组、谓词和 Hopper descriptor 寄存器对，以及保守内存/屏障顺序。LOAD 的首个寄存器使用距离为 d，则 exposure 为 `Σ max(0, W-d)`，W 固定为32或128。短距离可以获得非负风险费用；距离更长不会仅因为数值变大而被加罚。dependency depth 是单位边长路径，ready width、pending loads、首个浮点操作前 LDG 数仅是诊断，均不是 GPU cycles 或硬件并发度。

支持域是 `sm90-scalar-straight-line-native-v1`：精确选择的 SM90 入口、首个无条件 EXIT 前的受支持标量指令，无此前控制流、调用、异步/Tensor 操作或未知 opcode。EXIT 后 padding 不计入。编译器可把无控制流的源级除法变成异常分支与辅助调用；这类镜像明确返回 unsupported。未知操作、格式缺失、工作预算超限、工具失效均没有可评分的伪0特征。

组合评分把 shared 的读写字节合为 `shared_transfer_bytes`，local 的读写合为 `local_transfer_bytes`，避免实验不能独立辨识的重复列；再附加 Frisk `symbolic_bank_excess`，形成14列。bank 来自实际 packet 的独立 backend/observer 和 affine ownership 计算，读/写各自记录后求和；重发射 TTGIR 必须与已编译 TTGIR 字节身份一致，native LDS/STS 字节分别必须等于该物化通信的 load/store 字节，否则整池回退。bank 是地址证明负担，仍不称动态 native 冲突测量。Shared bank 冲突与 warp 地址分布有关，因此不能仅用 LDS/STS 总量代替。[NVIDIA Shared Memory](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#shared-memory-2)

缓存绑定 cubin SHA、精确 entry、加载时分析器源码 SHA、反汇编工具二进制与版本、schema，并核验 payload 摘要。profile 进一步绑定该提取契约，以及 bank 提取器、backend、独立审计器和 redistribution 四个实现文件 SHA。工具或实现变化不能悄用旧权重。

编译 REG/STACK/LOCAL/SHARED 作为反馈诊断保留，不伪装成编译前物理资源预测。NVIDIA 的资源表将 REG 记为数量，STACK/LOCAL/SHARED 记为字节；本轮仍将 LOCAL/STACK 字节、native local 指令、未知 spill 次数分别报告，并要求已有 driver 资源闭环确认。[NVIDIA cuobjdump Resource Usage](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-binary-utilities/index.html#cuobjdump)

## 4. 通信训练与正费用

通信训练有16个预登记 PTX 镜像：1个无通信基线；barrier、shuffle 各8/32/128次；Shared 各8/32/128次且 stride 为1/4/8。固定128线程和同一 uint32 输入协议，依赖链通过读写/XOR保持工作存活；测量包含10个配对批次，每镜像每批5次图回放，每图1024次调用。共160个 image-batch、800个 event；输入、输出和保护区均按协议核验。其数值/事件检查是训练事实，不替代正式留出所要求的四种 sanitizer。

Shared 的边际残差按同批控制计算：`(T_shared-T_base) - (B_shared/B_barrier)*(T_barrier-T_base)`，再用共同 batch 索引 bootstrap；没有把 barrier 的点估计当成无误差常数。三个重复长度的族统计给出正 lower CI，再取其10%作为保守校准先验。这包含保持工作存活所需的 XOR、依赖与地址协议费用，不能当作孤立硬件指令延时或任意完整内核的严格下界。

| native 校准先验 | µs/静态单位 | 静态 v6 对应最低 ticks |
|---|---:|---:|
| Shared transfer byte | 0.000314648469 | `shared_warp_issues` ≥315 |
| Shuffle instruction | 0.001609521496 | `shuffle_warp_issues` ≥403（4 warps/指令） |
| Barrier instruction | 0.000607617221 | `synchronization_barriers` ≥608 |
| 纯/通信地址 ALU | 未宣称测得延时下界 | 各≥1，为明确工程先验 |

固定4 warps 时，每条标量 Shared 指令为4字节/线程，对应4个 warp issues，故这两种标度可在本支持域内对应。不能把此单位映射外推到其他线程数或向量指令。

## 5. 训练规模、拟合与独立复核

所有旧的 R2/R4 性能留出在本轮均已转为已观察开发数据，绝不再次当新留出。静态训练重放旧完整赋值，逐个桥接原 domain、赋值、TTGIR/config/cubin 与旧标签，未重新搜索以替换训练镜像。native 训练只取明确支持的已观察图，不用不支持图的局部特征代替完整分析。

| 数据/模型 | 规模与含义 |
|---|---|
| 静态 v6 | 64个精确镜像标签，来自12个完整程序规格、24组；不是64个独立算法 |
| native 已观察训练 | 70个 group-image 中42支持、28明确 unsupported；支持标签来自8个程序规格、16组 |
| 通信训练 | 16个微基准镜像；不是16个自然图程序 |
| native 最终拟合 | 42+16=58行，97个组内 pairs；加弱绝对项后155×14设计 |
| native 设计秩 | 13/13活跃列；LOCAL整列未激活，仍不能证明参数具有普遍物理可辨识性 |

静态 v6 使用非负约束拟合及上述费用下界；训练绝对 RMSE 为14.9553µs。这一精度仍明显不足，模型本轮仅作为待验收的候选排序代理，尚未通过外推验收，不能据训练收敛宣称通用时间预测合格。

native 采用每程序组等 loss 质量的配对相对差量，另加固定0.05权重的相对绝对观测，去掉跨程序 launch/intercept 的主要干扰。约束求解在减去正费用下界之后进行，最后加回下界；系数以每微秒1,000,000 ticks 向上取整。最终 NNLS 11次迭代收敛，KKT≈2.85e-15。训练残差0.0514894是归一化、减下界设计上的数值，**不是µs MAE**。

原始 native 特征仍有多项拟合为0，例如总指令、global load、float/integer 计数和32窗口 exposure。这说明该有限目标选取了其他相关解释通道，不能称这些工作物理免费；最终模型仍依赖训练分布。未激活的 LOCAL 特征一旦在待评分候选出现，则整池回退，而不是零费用外推。

独立审计没有导入同一拟合器，而是从训练 ledger 重建97对差量、归一化和155×14目标，并穷举8192个活跃子集（1128可行）重求凸 NNLS。最优系数与实现最大差1.485e-15，费用下界与整数 ticks 一致。58行的 cubin、工具、源码/提取契约及整数评分也全部一致；工具×2、分析源码、bank源码、缺失证明及训练未覆盖 LOCAL 的6类内存变异均被明确拒绝评分。16个微基准的 PTX 模板、独立128线程 bank 枚举与通信费用复算一致。

`communication-training-r3-native-refresh` 只用最终分析器对原16镜像做CPU重提取，原15项 native 特征完全相等，原始计时文件逐字节相同，没有重新测量或择优换标签。静态 final 重放64镜像同样确认 features、赋值发射标签及 profile 字节一致。

开发历史原样保留：首次微基准 PTX 编译失败没有 GPU 工作；native 绝对拟合 r1 对程序规模过敏；r2 在正式新性能前被配对控制不确定性、bank 和提取身份契约替代；一次 CTest 中旧 bench 未全量重建已通过全量 Ninja 和重新验证修复。失败目录不是被新结果覆盖。

## 6. 已观察 exp/conflict 回归只能作为开发检查

上一轮 exp32×32 的未使用 V 读收费由 v6 whole-SSA 关闭；native 两镜像均只剩64个 byte loads，Shared 慢方案仍有额外96+96个 native Shared 声明字节及6个屏障。其慢/快主指令数1158/458得到保留，因此新反馈不再把 ptxas 删除的32条字节加载作为快方案的实际成本。

上一轮 conflict128×32 的慢/快 native load/math 总量相同，主指令1848/1829；32窗口 exposure 为1058/85，128窗口为20468/15968。保守 dependency depth 反而为533/612，说明不存在一个现成的“指令更多或链更长必然更慢”规则。新组合共同利用已观察数据校准，最终 native 评分在这两组都选择原先较快的镜像，开发方向正确。

上述标签已参与开发/训练，不能将这两个纠正后的排序宣传为新性能留出收益。训练也没有全对：R4 controlled diamond 的已选训练 median regret 仍约1.00949。是否改善新形状、自然候选池与外推，必须只看下述冻结协议之后的正式结果。

## 7. 六候选反馈预算与充分知情局部对照

每个程序固定六个来源：legacy/v5d/surviving 三模型各运行 LI 与 J-new。每个来源最多65536 probes、256 refinement；组合预算为393216 probes、1536 refinement。候选、编译请求及唯一 native 镜像上限均为 K=6。所有 prepare、solve、probe/refinement、compile/cache 和 native analysis 的消耗单独记账；这不是与单次65536-probe静态求解同成本的方案。

全体候选先完成共同算法、ABI、精度、合法性和 GPU 支持检查，再使用训练冻结的 native+bank 评分。任何候选 unsupported/unavailable、bank宽度对不上或激活训练未覆盖通道，整个程序统一退回 `surviving-J-new`。不能筛除难分析候选后宣称 native 排名成功。mixed/div 和三类 Attention 已预登记整池回退；diamond/remat/conflict/exp 是计划可覆盖的四类，若实际池出现不支持镜像仍遵循同一回退规则。

`native-LI-opportunity` 与 `native-J-opportunity` 得到完全相同的公共池、编译反馈和模型，故必须作出相同决定。这是信息/编译机会对等检查，不能据此证明 J 优于充分知情 LI，或把相同镜像的两个别名视为两次独立收益。与单搜索静态参考 `legacy-J-new` 的 GPU 时间对比还必须同时报告额外设置代价。若每次 kernel 的节省为正，可描述性计算 `额外设置时间/单次节省时间` 的摊销次数；没有设置成本测量时不填估计数。

## 8. 新留出验收结果

新六例完整程序身份与 case/tail/ABI/dtype/threads/N 均与所有历史规格去重。全部128线程、单CTA、相同算法/ABI/精度/候选域及已注册预算。Attention64×16三个ABI是已观察回归锚点，不进入主六例几何平均。

| 程序 | 形状 | N | 实际决策 | native-pool中位µs | legacy-J-new中位µs | 有限池regret |
|---|---|---:|---|---:|---:|---:|
| diamond | 256×32 | 8192 | native→surviving-LI | 11.20862 | 18.19123 | 1.000000 |
| remat | 128×64 | 8192 | 整池→surviving-J-new | 18.24933 | 18.24933 | 1.000000 |
| conflict | 256×32 | 8192 | 整池→surviving-J-new | 18.46354 | 18.46354 | 1.668361 |
| mixed | 256×64 | 16384 | 整池→surviving-J-new | 55.72000 | 55.72000 | 1.000000 |
| div | 128×64 | 8192 | 整池→surviving-J-new | 25.61993 | 25.61993 | 1.000000 |
| exp | 128×32 | 4096 | native→surviving-LI | 5.13375 | 6.05224 | 1.000000 |

表中µs为聚合中位数描述；正式比较和regret来自每个paired batch的比率中位数及区组bootstrap，不能直接用两列聚合中位数相除替代。自然图参考仅为固定六来源已编译镜像并集，不是候选全域GPU最优。

| 验收项 | 结果 | 门槛/结论 |
|---|---|---|
| 主六例时间比 | 0.900428329 [0.896961307,0.903004470] | 上界<1/1.03，通过 |
| Regret几何均值 | 1.089051266 [1.084817586,1.091463306] | 上界≤1.03，失败 |
| 最差regret | 1.668360678 [1.629822470,1.690654395] | 上界≤1.10，失败 |
| 六新+三锚点护栏 | 9/9通过 | 各alpha=.05/9，上界≤1.05 |
| 默认晋升 | false | 保持legacy；不开展外部公平对比冻结 |

两个实际 native-ranked 新程序中，diamond 相对 legacy-J-new 配对时间比为0.628304505，exp为0.848251677。其余四新程序回退所选镜像与legacy-J-new相同，时间比为1。不能把两个成功程序重新组成主样本或用其结果替代原六例。

### 8.1 功能、预算和测量完整性

- 54/54配置生成及GPU验证通过；9702条常规检查、2646组输入/参考/偏移；92个实际sanitizer进程（四工具各23）、324条安全子集记录，31项精确安全复用。复用不算额外执行。
- 36个f32发射覆盖通过，其中v5d计价18维12项、v6计价20维12项；18个Attention来源明确发射fallback，其中6个v6明确存活fallback。旧64个v6重放已另经完整强20审计。
- 单次最大10074 probes、256 refinement、73 seed work；9程序合计193906 probes/3656 refinement，每程序全部六来源均满足393216/1536总上限；18个已构造witness未丢失。
- 9组、23个group-unique精确执行身份、90个paired batches、230个image-block、1150个event样本；实际图节点4–759，event/count方程、随机顺序和输入pair SHA均核验。0 invalid、0 replacement，外部PID列表为空。
- 独立统计先于官方assess完成；468项聚合检查最大差1.11e-16，5256个单程序统计标量差0，全部门槛判断一致。

### 8.2 失败原因与下一项必要修复

native实际覆盖为2/6新程序。mixed/div和三Attention因异常控制流、MMA/异步等预声明域外行为回退；remat/conflict新增LOCAL激活训练零列，按预注册规则整池回退。所有候选均保留，未以支持状态筛掉慢候选。

conflict的三个不同执行镜像给出直接原因：

| 镜像/来源 | 中位µs | 物理资源 | 冻结native结果 |
|---|---:|---|---|
| legacy-LI | 71.44869 | GPR255，driver LOCAL4640B/thread | LOCAL训练零列激活，拒绝完整评分 |
| legacy-J-new/v5d-LI/v5d-J-new/surviving-J-new | 18.46354 | GPR80，LOCAL0 | 15,907,314 ticks |
| surviving-LI | 11.11177 | GPR80，LOCAL0 | 13,709,992 ticks |

冻结native模型已把两个可评分镜像排对，但一个未覆盖LOCAL费用的候选触发“任一未知即整池回退”，覆盖了这个排序。因此本轮主缺口是**不完整成本信息下的决策过于保守**。两份可评分镜像的GPR/LOCAL相同，不能把它们18µs/11µs差异归因为寄存器阈值。LOCAl字节也不等于spill次数。

事后仅做代数诊断：该未知LOCAL候选的已知非负项下界为46,864,408 ticks，已高于已知快候选13,709,992。在线性模型族中，任意非负未知LOCAL单位费用只会把前者继续加大。因此后续可实现带证书的部分成本区间：保留所有候选，对未知项形成[已知贡献,+∞]；只有所有未知候选的成本下界均不低于已知incumbent成本时才保留该incumbent，否则仍明确fallback。这个下界证书是条件模型族结论，不是硬件时延下界；**本轮未实现该运行时规则、未更改选择，也未据此重报性能**。证据为`development/conflict-partial-model-bound-diagnostic.json`。LOCAL费用校准与控制流费用覆盖也仍需完善。

### 8.3 搜索、编译与反馈开销

新路径当前是六来源portfolio，所有三次prepare、六次来源CLI求解、六次compile请求及native分析均计账；并非单次65536-probe优化器。本次是普通编译缓存的实际顺序观测，不是独立冷启动比较，不能把多个组成耗时之和叫并行墙钟。验证/sanitizer与计时装置开销另有原始记录，不混进kernel时延。

| 新程序 | Portfolio设置s | 同轮legacy-J组成s | 描述性摊销调用数 |
|---|---:|---:|---:|
| diamond | 149.907 | 15.134 | 19,301,317 |
| remat | 172.505 | 15.714 | 无kernel节省 |
| conflict | 158.370 | 15.694 | 无kernel节省 |
| mixed | 726.518 | 25.003 | 无kernel节省 |
| div | 133.617 | 16.373 | 无kernel节省 |
| exp | 93.896 | 11.550 | 89,653,655 |

六新程序portfolio设置合计1434.814s；九程序合计1612.235s，54次编译请求的观察耗时合计1096.297s。mixed legacy-LI约8.7MB PTX，compile+numerics worker约569.6s，在原600s超时内完成，未重试。

摊销数由额外观察设置开销除以聚合中位kernel节省计算，仅作描述；没有设置成本独立冷启动置信区间。当前不能证明单次/短程调用的端到端收益。需要在生产接入前把候选archive并入同一共享预算、控制编译成本，并完成未知费用的安全部分排序。

实际资源为GPR32–255、静态Shared0–32768B、driver LOCAL0–63896B/thread；10配置非零LOCAL。资源均来自编译器及加载driver反馈，spill次数和动态occupancy保留unknown，不把实际读取值称为静态预测MAE=0。

## 9. 冻结身份、已通过检查及剩余限制

| 身份 | 最终值 |
|---|---|
| 源码 recipe digest | `54e1a8ad263c07809291334c046efb32178fa26a76d0266ed227d942dc9646ec` |
| Recipe / archive / binaries | 417文件 /527文件 /5个二进制；patch与隔离重放通过 |
| Git HEAD | `9f79155ac08f82133f4ddb6554af552feefca8d2`；未提交本轮修改 |
| 静态 profile 原始 SHA | `9ee243bfa7845822857fa1efecc3f229563e78cb6ea314ef179e38e59400d2e6` |
| 静态模型语义 ID | `sm90-layout-calibrated-v6d-1850a5d94517c090e16c713c307017c4-leg` |
| native profile 原始 SHA | `0900c1a59ad70d577f4a7be7a0cbc47ea9d1588319faf4d0b9a8a9ee952c2a6c` |
| native analyser SHA | `bb536a9ba88e7d28a9cac60ae50c9d234dd5ad61241936454a30cc08114ddf07` |

已完成全量 Ninja；675项布局单元、48项选择/Oracle、29项focused、12组CTest及80项lit通过。native解析器16项语义/缓存测试通过；已观察R4的36个cubin中23支持、13明确回退，全部36的独立SASS opcode与REG/STACK/LOCAL/SHARED对账零差异。测试分组可有包含关系，不能把这些数简单相加当独立测试总量。

本轮的创新载体是可审计的组合：消费者敏感 GF(2) 查询、规范原像/所有者、全图布局与转换耦合、静态存活证明、通信正费用及有预算的编译反馈。GF(2)、像核空间、bank分析或成本反馈各自都不能被宣称为 Frisk 独占。本轮没有 Triton/TileLang 同算法、ABI、精度、搜索与编译预算的外部公平实验，不能宣称已经优于二者。

仍需保留的局限是：whole-SSA闭包不是完整ptxas优化模拟；native支持域不覆盖异常控制流、MMA及异步流水；静态绝对时间训练精度仍差，native归一化训练收敛不等于通用外推；bank是与native宽度绑定的静态地址证明而非运行时测量；LOCAL、动态occupancy及spill次数不靠线性代理虚构；公共池内的GPU质量不证明全域搜索最优，compile额外代价也不能省略。单CTA/4warps结果不代表多CTA扩展或完整Attention端到端收益。

该源码冻结是可重放的实验快照。当前新留出总体选择质量失败；默认保持legacy，不进入对Triton/TileLang的外部公平对比冻结。

## 证据导航（本阶段根目录下）

- `development/source-freeze-final/freeze.json`：源码、二进制、patch/archive与重放身份。
- `development/integrated-validation-final/validation-summary.json`：全量CPU验证。
- `surviving-known-training-r2-final/manifest.json`、`surviving-fit-r2-final/`：静态64镜像重放与最终profile。
- `development/final-static-replay-equivalence.json`：最终64镜像复验等价。
- `native-known-training-r3/`、`native-fit-r3/`：42支持/28回退的提取与最终14列模型。
- `communication-training-r2/`、`communication-training-r3-native-refresh/`：原始通信训练与无重测身份刷新。
- `development/native-analysis/observed-r4-native-analysis-final.json`：36个已观察native镜像独立对账。
- `development/native-analysis/calibration-review-final-nnls.json`、`calibration-review-final-binding.json`：独立最优解与绑定审核。
- `development/native-training-ranking-r3.json`：只限开发的已知排序，不是新留出验收。
- `development/development-history.json`：失败与被替代开发实验保留。
- `final-configuration-r2-final.json`、`run-native-cost-repair-r1-20261103/protocol.json`：最终输入身份、预算、支持域、预登记门槛。

## 最终验收证据

- `run-native-cost-repair-r1-20261103/quality-summary.json`、`native-group-analysis.json`：正式门槛及各程序数据。
- 同run `native-decisions.json`、`measurement-aliases.json`：计时前固定的选择与精确身份别名。
- 同run `measurement/{freeze,completion}.json`及全部raw：90paired batches，1150event样本。
- `development/root-gpu-validation-audit.json`、`root-measurement-audit.json`：独立完整性与实际执行计数。
- `development/independent-new-holdout-statistics.json`、`independent-final-statistics-review.json`：先独立后官方的统计复核。
- `development/independent-group-statistics-review.json`：单程序全部统计对账。
- `development/conflict-fallback-diagnosis.json`、`conflict-partial-model-bound-diagnostic.json`：失败因果边界及非运行时的后续证书。
- `development/old-sealed-preservation-final.json`：旧19873文件/6,518,516,949B和exact fileset保持。
- `evidence-index.json`：最终精确文件集、所有SHA及报告快照；该索引封存后不再修改其包含文件。
