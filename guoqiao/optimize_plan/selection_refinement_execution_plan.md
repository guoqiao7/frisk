# 任务1：候选保留与成本决策联合修复

本阶段只在 H100_direct 的 `/home/baopeihua/frisk` 修改源码。之前 network-selection R3 的所有程序、性能和原始证据均已观察，统一作为开发、训练与回归材料；不再称新留出。旧跨 HOME/DATA 索引 SHA256 为 `a666ac9cc5a987eebc44d431154dd1c0e46528966a5c2e0e8e7c131beefbc423`，只读。

## 范围与开发验收

1. 在相同 K4、总 65536 probe、256 refinement、1 个网络变体保留评分的预算内，修复已付费候选丢失。完整程序身份不可被总量特征向量替代。静态代理支配不能跨不同 GF(2) 通信切分结构无条件删除候选；实际 solver/local/static pins 保留。旧默认策略保留为对照。
2. 对旧 native14 不支持的有界 scalar call/CFG 与当前 Attention 指令族补独立静态费用合同；不把跨合同静态计数直接套用 native14 系数，不把静态完整分析声称为动态时间上界。
3. 在单位、源与镜像绑定明确的前提下联合 native 工作、通信和 GF(2) 地址局部性；保留精确有理可识别性诊断。删列需非负有理锥组合证书，且实际推断池重验，不凭训练矩阵线性相关删除带新正价含义的轴。
4. 模型选择以完整池、充分知情旧策略为 fallback。检验排序、默认实际选择及退化保护，不能将单纯 raw fit 改进或全部 fallback 冒充决策质量修复。共同成本的附加偏移不应任意改变对比证据；受控差分须证明其干预范围，不把计数相同当作执行相同。
5. 已知 exp128×64 检查候选是否被 paid 观察及保留；已知 div512×128 检查可识别信息与默认决策。其余 R3 21 程序、42 J/LI 池执行开发回归，记录每次失败/探索，不作泛化验收。

## 不包含的后续工作

全链路搜索/规范化/证明缓存与费用收敛属于任务2。本阶段只记录实际增加的分析费用，不开展新缓存工程。新的真正独立性能留出、默认版本晋升、Triton/TileLang 外部公平比较属于后续任务。没有新证据前默认 legacy 不变。

## 证据路径

- 小型开发收据：`/home/baopeihua/frisk/guoqiao/optimize_plan/selection_refinement_artifacts/development`
- 大型输出：`/data0/baopeihua-frisk-selection-refinement-20261105`
- 纯推断已知池回放：`benchmark/layout/execution/selection_refinement_decision.py`，可以安装禁止读取性能/训练 label 的进程 audit guard。
- 最终状态以 `guoqiao/inference_detail/selection_refinement_report.md` 的实际完成证据为准；本文件为事先范围与验收约束。

## 开发验收结果

本轮最终采用显式source-grammar gated recipe，已知Exp新J/LI闭环，Mixed/Div完整保持旧归档；强制全域GF2的Mixed 2.527×退化原样保留。完整最终边界和证据见 `/home/baopeihua/frisk/guoqiao/inference_detail/selection_refinement_report.md`。本阶段不是新留出或性能晋升，默认legacy保持。
