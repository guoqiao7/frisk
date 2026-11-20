## 2026-11-02 执行结果

本轮实验、GPU与证据审计已完成；整体模型/选择质量资格未通过，calibrated不晋升，默认保留legacy，尚不冻结开展跨框架对比。实际指标及模型覆盖/自然搜索缺口见[验收报告](../inference_detail/model_selection_quality_report.md)。原预登记计划完整保留如下。

---

# 模型及选择质量验收执行计划

登记日期：2026-11-02（Asia/Shanghai）。本阶段在 H100_direct 的 `/home/baopeihua/frisk` 执行，修改及证据仅保留在服务器，不自动提交或推送。正式来源、工具、协议与系数身份在运行登记时冻结；结果出现后不重新拟合、调整形状或放宽阈值。源码维持正式冻结状态，本文为阶段说明文档。

## 验收域与完整分母

新留出集按完整的图／形状／尾部计算／ABI／精度／线程数定义：diamond 64×32、remat 64×32、conflict 128×16、mixed 16×64、div6 32×64、exp1 64×32。六种完整程序组合均不在历史登记夹具的 23 个身份中；部分单独的尺寸曾在其他图中出现，不能称为所有维度均未见过。Attention 64×16 的 weak-vector、strong-cpasync、strong-tma 三个 ABI 为已观测锚点，单列报告。

总分母为 **84 个配置**：六个新程序分别具有八个受控配置及四个自然配置，共 72 个；三个 Attention 锚点分别只有四个自然配置，共 12 个。受控八配置为 legacy／calibrated 分别执行 forced-A、forced-B、LI-C、J-C；自然四配置为两个模型分别执行 LI-C、J-C。所有不支持、选择失败、导出失败、编译失败、资源拒绝、测量无效均保留，不以替代样例补足分母。

受控域具有两个 producer 选择，其余候选为单例；未强制 producer 的 LI/J 共 **24 个配置**执行精确 oracle 检查。J 检查完整模型最优解，LI 分别核对充分知情的局部第一阶段及固定 producer 的完整第二阶段。forced A/B 仅用于两候选枚举及实测参照，不额外计为独立 oracle 验收。

自然域使用实际初始化的候选，两个模型及 LI/J 对齐冻结 domain／coverage 身份和预算。自然域实测参照是 **四个已选配置的可执行镜像并集**，不枚举整个原始候选空间，不能声称取得自然域的模型全局最优解或完整候选实测最优解。相同镜像的多个名字继续保留，不计为多个独立样本。LI-C 纳入 producer 自身算术、访问和必要输入修复成本，不读取未来消费者；本阶段只比较 C，已有 LI-O 全域限制不纳入比较。

## 模型、预算与公平条件

legacy 使用 `sm90-layout-lex9-legacy-v2`，calibrated 使用具有冻结 profile 哈希的 calibrated-v3 语义身份。系数文件仍采用 `sm90-layout-calibrated-v2` 格式，SHA256 为 `6b4947af82c2748fc8db75268ddb621fc45852d238b63deee12cf17c6cde7388`；本阶段不拟合。

保持同算法、ABI、精度、候选域和搜索预算：exact combination limit 256、beam 32、max expanded states 65536、启用 beam refinement、max refinement probes 256。GPU 为 UUID `GPU-028d5ad6-2ff8-f6e3-130f-9eefb0447618`，SM90、单 CTA、128 线程、4 warps、`enable_fp_fusion=False`。O 与 Q/K/V 有效区域不相交，只读输入可互相重叠。不同 Attention ABI 不通过跨 ABI 比值归因布局收益。

所有登记候选和选择结果分别完成正确性及物理资源闭环。普通 f32 对应操作逐位一致（包含 signed zero）；exp1／mixed 沿用有限误差 `abs(error)<=1e-5+2e-6*abs(reference)`；Attention 沿用 `.002+.002*abs(reference)`。输入、独立参考、地址偏移、stream／graph、守卫及四个 Compute Sanitizer 工具沿用已验证闭环规则。安全检查只有在 cubin、完整 emission config、物理 ABI、完整 spec 和安全输入协议均一致时才允许复用。

## 计时与判定

每个程序有十个完整成对批次，warmup 100、pilot 100 nodes、目标 graph 5 ms、graph nodes 范围 1..4096、每批五次 replay。f32 第 i 批沿用 `calibration_fixture.inputs(spec,i)`（seed 为 20261031+i，i=0..9）。每批按精确 cubin／config／ABI 身份去重计时，同时保留所有强制及所选配置别名。记录所有编译、计时及拒绝开销。目标 GPU 出现外部进程则整个成对批次失效并保留，不替换；不操作其他 GPU 的外部作业。

统计以完整成对批次为单位，10,000 次 bootstrap、seed 20261102；重复 event 或镜像别名不增加样本数。给出实际选择的 empirical regret 及区间，受控参照为两种 producer 的完整有限集合，自然参照为四个所选镜像的并集。候选排序使用镜像对的 Bonferroni 区间和 3% 实际差异阈值；无法区分或同镜像应报告平局／未决，不视为性能提升。

有限验收域的 regret 门槛为几何平均不超过 1.03、最大单例不超过 1.10，并区分通过、失败和区间未决。calibrated J 在受控及自然两套中的 regret 均为默认晋升前提。calibrated 默认晋升的主比较仅使用六个新留出程序的自然 J：calibrated/legacy 几何平均耗时比的 paired 95% 区间上界须小于 `1/1.03`；三个 Attention 锚点不进入主聚合。另要求六个新程序加三个 Attention 锚点全部通过单例退化门槛：耗时比上界不超过 1.05，以九次比较 Bonferroni `alpha=.05/9` 控制族错误。还需完整覆盖、同条件和无硬资源违规；未满足则继续采用 legacy。

J 与充分知情 LI 的实测比较允许零收益或负结果，不以成本差代替 GPU 收益。固定样例 bootstrap 区间不代表新程序总体的泛化区间。该阶段完成可以得到“calibrated 未晋升／legacy 保留”的结论，不修改门槛追求预设胜负。

## 资源与证据边界

pre-native Shared 结构预测需与真实编译及加载资源核对；寄存器使用真实编译器及已加载 CUDA driver 反馈，不恢复已经撤回的静态物理寄存器点预测。记录 local bytes；没有显式原生 spill 元数据时 **spill_count 为 unknown／None**，不能由 local bytes=0 推断 spill=0。资源硬限制不能通过删除失败候选绕过。

广泛延迟外推及纯静态物理寄存器预测仍未 qualified。本阶段只验收冻结有限域的模型排序与实际选择质量，不声称完整自然候选空间最优、任意程序外推准确或已胜过 Triton／TileLang。

新证据全部位于 `guoqiao/optimize_plan/model_selection_quality_artifacts/`，与旧阶段隔离；旧报告、执行计划和 evidence-index 不修改。独立审计六个旧封存索引 SHA256 及索引内位于各自阶段目录的所有条目 SHA256／bytes，写入本阶段 `development/prior-sealed-audit.json`。旧索引对当前源码及长期文档的引用不属于“旧阶段目录内容保持”断言。
