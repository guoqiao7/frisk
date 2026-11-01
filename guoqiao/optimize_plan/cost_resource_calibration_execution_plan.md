# 成本与资源校准阶段执行计划

日期：2026-10-31。用户授权：在服务器实现成本与资源校准，同时完善成本模型并纳入充分知情的局部基线。作为一个阶段交付，内部连续实现、验证、复审；不另拆用户任务。

## 基线与范围

- 服务器 `/home/baopeihua/frisk`，HEAD `dd577f6c1e0c87fbe5b5bf204fd250c00a613ce1`。前一阶段工作区修改已经验收但未提交，本阶段保留，不自动暂存、提交或推送。不保留本地代码副本。
- `cost_resource_calibration_artifacts/stage-start.json`记录起点及源码哈希；核心preimages保留服务器。上一阶段的报告与7632项证据索引不覆盖、不重写。
- 继续使用同一受检Frisk→Triton低层后端及H100 GPU1。禁止将实际GPU模型代理量改名为物理寄存器、DRAM流量或硬件周期。
- 不包含搜索开销优化、跨框架性能结论、任意形状/循环/流水调度扩展。

## 内部实施与验收

1. **统一可选v2模型与充分知情LI。** 默认legacy不变，显式profile可从正常推断入口和实验入口使用。使用非负整数tick校准评分、稳定模型/参数身份；加入普通算术、独立除法/exp/cast及route-specific转换、同步特征。所有转换/重计算方案用同一模型比较，最终方案携带可复算特征账本，拒绝方案不污染贡献。代价仍可加；下界先保守为零，不以非可加全图峰值破坏component求解。未知操作/配置明确拒绝或标记未覆盖，不默认为免费。
2. **充分知情局部基线。** 保留旧L，新增LI与J。在完整合法域上，LI第一阶段计入声明producer操作自身算术/指令/访存/路径/直接资源贡献，允许其本地辅助Storage选择；排除未来consumer转换与重计算。混合未来消费者的非局部资源聚合不可静默纳入。第二阶段仅固定producer输出encoding及geometry/path，不额外锁死consumer Storage。共享预算、稳定tie、独立小域oracle及pin语义保持一致。贡献归属/依赖可复核。
3. **训练、冻结与留出验证。** 按整图和shape预登记train/holdout，拟合超参在观察holdout前固定。训练覆盖受检机制图、算术链及Attention路径，留出使用不同图/shape/混合运算。未激活和共线特征明确报告未单独识别，不宣称通用时延模型。先独立数值及精确cubin安全检查，再正式配对计时；profile训练后冻结，holdout不反向调参。真实运行legacy/v2下LI/J选择，报告候选排序、近噪声、选择regret及退化，不能只展示分数降低。
4. **资源校准。** 在实际物化/后优化IR采集既有SSA峰值和Shared roots诊断；当前loaded CUfunction查询真实regs/local/shared及CUDA driver理论residency（不是实测occupancy）。按相同train/holdout分割评估逻辑代理对真实资源的预测误差，说明转换临时值、后端复用、packing等缺项；实际Shared须匹配emitter分配含scratch/barriers。暂不将非可加峰值直接替换可加搜索目标。
5. **整体交付。** 相关C++/Python/正常pass集成测试、适当完整回归、独立源码及数据复核；一个中文实施报告、一个证据索引和长期文档状态更新。明确校准支持域、不可识别参数及性能限制。未测得提升/明显退化均保留，完成定义不以取得正收益为条件。

## 测量纪律

- GPU1 UUID `GPU-028d5ad6-2ff8-f6e3-130f-9eefb0447618`；查询设备及foreign process并failclosed，不终止外部作业。
- 编译/JIT/加载/setup/capture与正式kernel时间分列；配对输入、offset、随机执行顺序、批次统计及置信区间口径预登记，不将graph内部launch当独立样本。
- 正式计时前冻结source/tool/spec/profile身份，期间不并行构建或运行重CPU验证；阶段间独立做source guard。不因结果不理想更换样本或调参。
- 所有新证据仅保存 `guoqiao/optimize_plan/cost_resource_calibration_artifacts/`，历史失败保留并标明，不冒充成功。
