## 2026-11-03：需求覆盖与同预算搜索复验（整体未通过）

本轮执行覆盖与预算内搜索修复已经实现并复验：114配置、16002数值/保护记录、144次实际sanitizer全部通过；36次受控Oracle和预算核对通过。原始计时150个配对批次/1800事件，无无效或替换批次。

整体模型及选择质量仍未通过：组合主时间比0.940789，但exp32×32退化44.9%；固定新搜索时，新成本模型反慢6.38%。自然域regret和单例保护失败，默认保持legacy-v2，不进入公平对比版本冻结。收益主要来自搜索修复，不能称新模型已经取得性能优势。新模型CPU solve合计约为legacy的9.25倍，仍需优化。

完整报告：`guoqiao/inference_detail/demand_cost_search_repair_report.md`；证据与保留的CPU失败历史在`guoqiao/optimize_plan/demand_cost_search_repair_artifacts/`。新性能标签只在冻结后的R4读取，没有调权或换case；以下原文逐字节保留，作为历史状态。

---

# 需求执行覆盖、同预算搜索与新留出复验

日期：2026-11-02。实现和证据只保留在 H100_direct 的 `/home/baopeihua/frisk`；不提交或推送。旧封存阶段的报告、索引、原始结果保持不动。

本文件是实现与验收计划，尚不是完成声明。最终注册需在实现、开发期模型/权重选择和 CPU 能力预检结束后，冻结真实模型 ID、profile 字节 SHA、feature contract、search contract、完整源码、工具及全部规格。

## 数据隔离

历史登记去重29个完整程序身份，包括上一轮六个新留出，它们如今全部为已观察数据。扫描全部11份历史 protocol.json 未发现额外完整身份。本次已知训练固定采用上一轮封存34个实测镜像（31个f32、3个Attention），对应9个完整程序，全部保留不挑选。上一轮六个失败/诊断程序已经观察，可以同时用于训练和覆盖/搜索回归，绝不能再次称作新留出。通过17-unitprofile完整赋值重放提取新特征，要求实际TTGIR/完整config字节与旧镜像一致、原cubin精确复用、native工具未变、旧timing/输入参考/ABI/hash链完整；不执行新的GPU训练。

新留出六例不仅完整 case/shape/tail/ABI/dtype/thread 不重叠，还要求同 case/tail/ABI/dtype/thread 的总元素数 N 未观察，防止点算子仅改变二维 shape 而复用相同 flattened 执行行为：

| 程序 | 形状 | N |
|---|---|---|
| diamond | 128×32 | 4096 |
| remat | 64×64 | 4096 |
| conflict | 128×32 | 4096 |
| mixed | 128×64 | 8192 |
| div6 | 64×16 | 1024 |
| exp1 | 32×32 | 1024 |

Attention weak-vector、strong-cpasync、strong-tma 固定QKV64×16作为三层已观察回归锚点，不参与六例主收益汇总。上述候选可以在正式注册前做 CPU 生成/发射能力预检，不能在看过最终 GPU 性能之后更换失败用例、调整权重或门槛。任何 unsupported 均保留并修复能力，不选择性删除。

## 单一维护的正交实验矩阵

使用共享质量驱动的显式 suite/model/profile/feature/search 参数，不复制整套测量逻辑或用 monkeypatch 改旧 fixture。完整注册明确指定两个成本模型（legacy baseline 与新需求语义模型），以及两种全图搜索（old beam、new witness）。

自然域每个模型分别 LI（witness off）、J-old（off）、J-new（on），共6臂。受控域再加各模型的强制 A/B（witness off），共10臂。六程序的受控+自然共96配置，Attention三自然域共18配置，总114；15测量组，受控36份未强制 LI/J oracle 记录。

主比较为 demand-J-new / legacy-J-old；固定搜索比较 demand-new / legacy-new 与 demand-old / legacy-old，固定成本模型比较 J-new / J-old。不得把两者一起变的总收益全归于成本模型。LI仍为充分知情局部基线，其信息包括 own producer arithmetic/access/input repair，不知道所有未来消费者。

所有臂保持同算法、ABI、精度、候选域与合法性证明；同一原生编译配置。总展开上限65536、refinement256、beam32、exact256；新 witness、构造/验证 incumbent、beam、refinement 全部在同一预算内记账，不能隐藏第二套完整局部搜索。

## 验收顺序与证据

1. 需求覆盖：先在已观察开发集对齐 C++ 解析 GF(2) affine-query DAG 与实际后端查询/边界账本。至少核对每SSA查询数、load/arithmetic、转换完整 source/destination vectors、barrier/Shared、memory sectors/banks 与地址生成费用。支持域外显式 fallback 原模型并记录原因，不得把 unknown 记零。若后端有可选 observer，应以其实际128线程查询向量独立重建 affine signatures，逐SSA比集合而不只比总数。
2. 搜索：复核实际构造的合法 incumbent 永不被更高完整模型成本覆盖；受控有限域仍由穷尽 oracle 验证；自然域报告已知可行下界和 beam gap，不能声称穷尽。成本改善不等于 GPU 必然改善，不能把前一轮 calLI 直接作为后验GPU赢家替换J。
3. 新留出：冻结一切后执行114配置准确cubin/entry/UUID校验、独立数值/保护、非默认stream/graph、四种sanitizer和编译器/驱动物理资源确认。Shared计划必须相等；寄存器用编译/驱动反馈，spill未知时保留未知，residency仅容量。共同demand编译查询预算65536（cache+pending），这是编译/代码体积约束而非硬件非法；old/new同一执行可行域。
4. 精确镜像/entry/config/physical ABI去重计时并保留策略别名，每组10配对批次、5event、warmup100、pilot100、目标5ms、最多4096节点。整批外部PID污染保留且不补样。冷编译、实际编译缓存、加载、描述符、搜索与图构造成本单独记录。
5. 原门槛不放宽：有限参考 regret几何平均≤1.03、最坏≤1.10，上界通过/下界失败/其他未确定；主收益上界需<1/1.03，九自然案例退化保护上界≤1.05、familywise alpha .05。所有模型及搜索消融完整列出，Attention不掩盖新留出失败。

绝对时间代理另行诊断：使用冻结 NNLS intercept 与17个系数，不对新留出拟合，不新增或放宽门槛。六个新程序与三个已观察 Attention 锚点分开，controlled/natural 单列。同一 group-image 的策略别名不重复加权；若不同完整赋值产生不同预测，全部保留并先在镜像内平均绝对误差，再跨镜像报告 MAE，不能挑选最接近实测的预测。intercept 仅用于时间诊断，不进入选择目标。

如果拟合任何权重，只能在已知 training/development 程序中进行，并在 final holdout GPU时间出现之前冻结。新模型/搜索即使修复已知反例，也必须由新留出达到原门槛；不以“实现完成”代替质量通过。

## 创新与借鉴边界

借鉴Triton的是“只为变换后存活和新增工作计价”、资源容量检查、预算内局部保底及少量硬件实测等通用工程原则。GF(2)、核/正交补、子空间交、广播消除和bank计数已经被Triton等使用，不能作为Frisk独有发明宣传；也不复制其32/8等相对成本常数并称作新的模型。

Frisk这次要实现和核验的是：根据已选消费者布局与所有者，跨SSA反向传播 thread→register 的 affine query；按SSA+规范仿射查询去重实际复用，并保留不同查询的执行重数；对有界跨线程转换精确保留原通信及同步；把这些需求成本放进图级联合选择，与后端的独立执行账本逐项对应。这与只删除零寄存器basis、仅为局部重计算切片评分有明确任务范围差异。

上述范围仍受单CTA128线程、f32纯DAG、单最终store、C策略、有界query/cross边界支持域限制。实现事实、数学性质、已测收益和未证实的更广优势须分开陈述。最终若未达门槛，保留默认和失败证据，不强行宣布优于Triton/TileLang。
