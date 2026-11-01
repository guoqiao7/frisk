# 资源预测深入归因与修复：统一实施计划

日期：2026-10-31。用户授权在服务器深入研究资源误差并修复。仅 `/home/baopeihua/frisk` 修改与保存；不暂存/提交/推送，不创建本地副本。前一阶段已封存目录、报告及索引不修改；本阶段365份原起点源码前像与5份未改HEAD输入补证组成370份基线，最终完整源码包380份；五文档原文前像一起保存 `resource_prediction_repair_artifacts`。

## 目标和责任边界

1. 查清逻辑SSA代理、lowering临时值、ptxas分配、Shared scratch/对齐/barrier及工具报告的不同含义，使用可复放因果对照而非凭相关性归因。
2. Shared改为native编译前的结构分配计划，实际emitter和资源模型消费同一受检计划，并由独立packing oracle、native与driver交叉核验。
3. 撤下默认SSA线性回归的physical register点值；提供有总工作预算的typed PTX CFG压力（不是physical上界）。准确寄存器数量使用编译后CPU反馈，并以实际loaded image SHA/entry和CUDA driver确认；缓存绑定cubin/tool/schema/entry，变化或损坏failclosed。不将已编译反馈标成预测。
4. 正常编译/执行和默认资源报告接入新分层，旧回归仅显式legacy、标记unqualified，C++搜索期加性logical-slot风险代理保持其原先代理语义，不宣称其也变精确。

## 预登记验证

登记文件 `resource_prediction_repair_artifacts/prospective-registration.json` SHA `b02e0ada1cf04079c67aa3012b1a45319f82c613190c1be1c241b83eba79d6eb` 在任何新prospective资源观察前冻结。旧12spec全部作known（含上轮3个留出），新6spec为diamond16×64、remat32×64、mixed128×32、div6 64×64、conflict256×8、exp1 16×128，各A/B×C/O共24新候选；known42，共66。不得按结果换spec、补样本或调整阈值。

- Shared结构预测对supported新case要求0B误差；不能输入driver/cubin资源真值。
- 寄存器纯physical点模型若有，原登记MAE≤8与最大低估≤16；当前研究决定不发布该点估计，必须保持coverage=0/MAE=None/not-qualified，不伪称达标。
- compiler/driver register、结构Shared与driver需一致；raw cuobjdump SHARED与用户静态Shared、device-reserved必须分别列出并验证。STACK/LOCAL非零不猜spill或local总量，保存driver确值。
- 新66产物精确cubin数值、安全检查及身份闭环；安全复用仅raw cubin+完整config+ABI相同，不能按SASS同就复用。GPU1外部作业不停止、不干扰。本阶段不测kernel性能。
- CPU因果probe改变优化等级或将div换mul只用于归因，不纳入正确性/性能比较；标明与生产设置不同。

## 流程与最终交付

代码源审查关闭Important后冻结source/tool；新run必须在本stage根下并用绝对路径绑定。正常入口首先写execution-only Shared计划，再写PTX-only信息估计（Triton combined compile后、首次resource feedback前；不谎称墙钟早于ptxas），然后查询编译资源、实际加载并确认driver。阶段文件集合/注册/模型null状态完整哈希守门。

实施后运行适当Python/既有后端回归和CTest；全部证据在同一stage目录，最后一份中文报告、一个证据索引、五长期文档前缀更新、独立源码/数据/最终闭环复审。不因下撤未验证预测而宣称通用寄存器静态预测成功，也不把硬件资源准确反馈等同于跨框架性能优势。

## 实施验收记录

2026-10-31 完成：known42与6新spec/24候选共66配置资源、数值及安全闭环，Shared真实编译前结构预测误差全0B，compiler/loaded driver反馈一致；物理寄存器点覆盖0/24，保持未通过资格。136项Python兼容回归、显式backend6/export3、CTest10/10通过；216次sanitizer实跑、12配置按raw image/config/ABI精确复用，4674条检查通过。源与数据独立审查0Critical/0未关闭Important。

测量期371份source与最终tree仅`test_backend.py`夹具修正有差异，manifest不回写，双身份和前像已保存；生产实现/工具/接受产物不变。最终源码差异、五长期文档原文后缀及总证据索引另作独立终审。通用纯静态寄存器模型与性能收益不在本轮成功声明内。
