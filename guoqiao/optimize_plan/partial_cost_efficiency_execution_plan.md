# 部分成本决策、LOCAL校准与搜索编译开销优化

只在H100_direct服务器修改代码，保留此前三个sealed阶段。不提交，默认legacy不因开发回归自动改变。

1. 决策：完全可评分候选保留精确score；仅满足严格合同的training-zero且coefficient-zero未知列形成非负可加模型族区间[已知贡献,+∞]。只有所有未知下界均不低于精确incumbent时才认证该argmin，否则明确fallback。普通工具/架构/CFG/bank合同不支持不伪造界。证书绑定全部候选、profile、实现和分析证据，不删除候选。等号仅证明argmin成员，不虚称唯一。旧conflict仅作已观察回归，绝不重报为新性能留出。

2. LOCAL：用56个已观察可native解析镜像、16个旧通信微基准、22个新LOCAL微基准，共94个group-images校准。新LOCAL直接PTX动态索引thread-private数组，256/1024/4096words，8/32/128及部分512/1024/2048重复，匹配BAR+寄存器协议对照，CPU先验证实际LDL/STL/直线native。GPU1 normal checks+四sanitizer+10paired batches×5event/1024实际graph nodes，输入/输出guards和uint32参考。只拟合LOCAL一个非负系数并加独立正协议增量prior，旧13系数冻结；allocation bytes、static transfer bytes与spill次数分开。完整14维重拟合只作开发对照，不用于新留出选择。

3. 搜索：单v6模型一次prepare，协调producer-local见证与J，global65536 probes/256 refinement全流程共享不reset；local probe份额1/4，refinement允许用全局上限，J用实际剩余以保留充分知情局部见证。K默认4、硬cap6，在线bounded完整候选归档，强制保留实际LI/J incumbents，再在通用2×主cost前沿选少量完整assignment多样候选。精确assignment去重，不按feature向量去重；完整候选重新证书验证/零probe重放。可选observer和CLI，不开flag旧路径不变。另有同65536/256/K预算的fully-informed local archive对照。

4. 编译：单批次CompilerSession，一次Python/Triton导入和工具/源码context，key绑定完整TTGIR/config/target/options/toolchain/code-affecting env。精确重复只编一次，坏cache拒绝、单writer原子发布，hit按当前execution重建资源证明。compiler API调用与实际ptxas/internal cache hit状态分列，后者未知保留unknown。GPU严格执行通过compiler_provider只验证已编译receipt，镜像identity包含receipt，复用既有数值/safety/graph引擎。

5. 质量：原R1六程序现为观察训练/回归，不再fresh。预先固定新的完整程序及case/tail/ABI/dtype/threads/N去重：diamond128×128、remat256×64、conflict128×128、mixed512×64、div128×128、exp128×64；另三Attention为观察ABI anchors。CPU capability preflight允许并完整保留，不能在看到新时延后改权重/shape/gates。主六例regret upperGM≤1.03、uppermax≤1.10；joint-partial-local/legacy-static时间比upper<1/1.03；九case(alpha.05/9)guard upper≤1.05。same-budget fully-informed local/native单列，旧strict→partial→LOCAL增量在同joint池消融，无额外GPU样本；参考仅为注册archive+legacy已编译并集。

6. 开销：shared预算cap65536/256/K4对比旧393216/1536/6，同时记录所有实际probes、prepare、archive/replay、cache/API/compile/analysis/startup时间。必须在相同已观察R1输入上分离cold与warm比较；旧cache观测不能直接冒称新cold优势。冷测试使用独立Triton/exact-artifact cache，工具/代码/编译选项同一冻结源；warm只读同批命中并验证正确。验证/sanitizer和kernel时间分列，不用kernel收益掩盖设置成本。没有达标则保持默认并报告失败，不通过删例或改阈值获得通过。

研究与创新保留Frisk消费者敏感GF(2)需求、规范所有者、图级联合布局转换、执行对账；新增条件成本族证书与共享预算编译反馈。GF(2)/NNLS/基本lower bound本身不宣称独占。未完成Triton/TileLang外部同算法/ABI/精度/预算实验前不宣称优于它们。
