# 支持域与当前 GPU 闭环：实现与验收报告

日期：2026-11-02（Asia/Shanghai）。状态：当前预登记支持域完整执行通过，源码/数据独立审计通过；最终整体审查见本阶段 review/whole-stage-review.json，统一清单见 evidence-index.json。工作仅在 H100_direct:/home/baopeihua/frisk；Git HEAD仍9f79155ac08f82133f4ddb6554af552feefca8d2，未提交/推送本阶段修改。

## 交付结论与支持范围

11个自然程序/形状/ABI × legacy-v2和显式profile calibrated-v3，共22配置全部完成：正常推断 → 物化selected IR → 受检执行导出 → 实际TTGIR生成 → native编译 → 精确cubin加载 → GPU数值/安全/资源核验。没有A/B候选注入、形状替换或丢弃失败行。四个calibrated大图原先自然选Eall后导出失败，现在保持原布局执行成功；原五个成本失败配置所对应程序均在本矩阵内完成GPU闭环。

本轮固定SM90a/H100 GPU1 UUID GPU-028d5ad6-2ff8-f6e3-130f-9eefb0447618、单CTA、128线程、四Global根Q/K/V/O、f16/f32的受支持操作/合同；外部主矩阵为连续行主序。weak-vector根alignment1，strong-cp.async/TMA的QKV128/O1独立分层。两个模型均使用normal pipeline默认legacy boundary-frequency，不能据此称static-trip-count-v2也经过本轮GPU矩阵。显式calibrated-v3仍使用原冻结v2系数文件（raw SHA6b4947af82c2748fc8db75268ddb621fc45852d238b63deee12cf17c6cde7388），未重新拟合。

推断可评分/证明、导出、受检emission、native可加载以及GPU已验证实例，是五层不同声明。版本化边界详见 ../optimize_plan/current_gpu_closure_artifacts/support-domain-contract.json，逐例结果见 ../optimize_plan/current_gpu_closure_artifacts/validated-support-resource-matrix.json。支持查询成功只表示实际checked emission，不能替代后续native/资源/数值/安全步骤。

## 真实缺口与实现

开发预检22/22推断成功、18/22导出/生成成功；四个calibrated大形状自然选中Eall，首先被转换adapter寄存器上限拒绝，后续还有holder表的65536枚举边界。失败原始输入、selected IR、stderr保留development/natural-preflight。

1. **分阶段支持与拒绝。** C++导出诊断增加stage/code/proof并保留原因和原子发布；Python execution_support.assess/assess_file验证实际packet与SSA envelope，再调用真实backend，明确supported/unsupported/invalid/infrastructure边界，不用22实例白名单当能力规则。旧schema1和低层测试API保持兼容。
2. **紧凑GF(2)导出。** 大holder映射使用schema2保存canonical位矩阵行、硬件/逻辑位宽和拓扑。覆盖与replication以符号证明确认，logical(h)惰性查询；first_owner只保存Nlogical大小的表。逻辑坐标bitpack与row-major slot明确转换，不提升旧证明枚举预算。
3. **大源Register转换。** 新planSymbolicRegisterRedistribution只给export使用，固定physical thread后在源register像内求最小register，只访问有限目标holder。source_slots不展开，Python逐目标复核slot、同线程和最小本地owner。旧通用Shuffle/Shared规划器及独立test-lowering边界保持不变。
4. **按需标量化。** 大寄存器f32纯SSA DAG、唯一最终Store且转换均Register时，按每线程所需source slot反向生成load/算术；查询缓存仅复用同SSA定义/同查询。保持selected布局、算术舍入、source owner与线程内读前写顺序，不跨线程重计算取消Shuffle/Shared。所有需求先算完再Store；运行ABI另行要求O与只读输入区域不重叠，不能把线程内顺序冒称in-place跨线程安全。
5. **统一执行driver。** 新support_closure.py/fixture复用既有backend、compile_exact、精确loader、独立参考、资源确认及sanitizer基础；每层有身份ledger，report检查完整22笛卡尔积、嵌套文件哈希及安全复用来源。

例如diamond Eall仍声明6152个逻辑register vectors，后端只需56个消费查询；mixed Eall声明32800，只需288个消费查询。这些不是物理GPR计数，也不是无编译寄存器预测；真实物理资源见下表。

## 预登记数值、安全与资源验收

机制图为3个固定随机seed加zeroQ/constant/signedcoords/cancellation，共7输入，每个算术独立FP32舍入；mul/sub以uint32位比较（包括signed zero），mixed沿用abs<=1e-5+2e-6*abs(ref)。Attention使用原15输入、FP16输入、FP32逐K参考累加、稳定Softmax、P的RNE-f16转换、FP32 PV，结果有限且abs<=0.002+0.002*abs(ref)。没有按结果放宽阈值。

weak输入/输出offset1/2/3；strong QKV offset0/128/256与O offset1/2/3做9个独立组合。每例三次普通运行；全部机制输入及Attention seed0额外验证非默认stream、精确3-node CUDA graph。输出sentinel/前后保护、输入字节不变、uint64指针范围及O/input不重叠逐次验证。

正式22/22全部通过，3654条数值/保护区/stream/graph/runtime ABI记录完整；独立审计重建966组输入/参考SHA。安全工具为memcheck/racecheck/synccheck/initcheck，共60次真实进程（15个精确身份×4），日志零error/零hazard；另外7个配置仅在cubin字节、完整config、physical ABI、spec和安全输入协议全部相同时复用，普通数值仍各自执行。真实安全subset记录228条。initcheck的global初始化检查不能冒称穷尽Shared/WGMMA异步安全；仍依静态Copy/WGMMA/barrier合同与组合动态检查。

所有22配置的compiler与loaded driver寄存器/user Shared/资源可加载性一致，local bytes均0；spill_count保持null，不从local0或rawLOCAL0推出spill0。raw Shared、user Shared和device reserved分列，PTX压力/逻辑槽位不当作physical GPR。SASS及raw资源绑定各自精确cubin。

| 自然程序/ABI | legacy regs/thread / user Shared | calibrated regs/thread / user Shared | GPU |
|---|---:|---:|---|
| diamond-32x32 | 32 / 4096B | 32 / 0B | 两模型通过 |
| remat-128x16 | 32 / 0B | 32 / 0B | 两模型通过 |
| mixed-128x32 | 168 / 16384B | 48 / 0B | 两模型通过 |
| conflict-256x8 | 88 / 0B | 32 / 0B | 两模型通过 |
| diamond-32x16 | 32 / 0B | 32 / 0B | 两模型通过 |
| remat-32x16 | 32 / 0B | 32 / 0B | 两模型通过 |
| mixed-32x16 | 32 / 0B | 32 / 0B | 两模型通过 |
| conflict-64x8 | 32 / 0B | 32 / 0B | 两模型通过 |
| attention-weak-vector | 64 / 6144B | 64 / 6144B | 两模型通过 |
| attention-strong-cpasync | 64 / 6144B | 64 / 6144B | 两模型通过 |
| attention-strong-tma | 64 / 6168B | 64 / 6168B | 两模型通过 |

资源变化不等于kernel加速；本轮未做GPU计时或跨框架性能比较。编译为22次API调用，使用普通compiler cache，不能称22次cold ptxas编译。

## 身份、失败记录和回归

接受运行 ../optimize_plan/current_gpu_closure_artifacts/run-current-gpu-closure-20261102T081806Z：379文件GPU recipe，source digest 12708d8b26a38f7376def24d55640e78d4947c689f7ecc6b7b395b4c48e914a7；构建与完整源码归档394文件，recipe是其子集，额外15文件明确列在清单。protocol SHA673a3bb0fbff8871259189d95228028d11502cfbd9aad21dc7f5efe66c2980e1，registration SHA72bb1a16ac7b42a02a4209047c2fdccbe708e6e980301e2cb209423fd02ee5b5。FriskOpt SHA143a3cd63e384394b6ed6fe3a7868d635f87c0ebe64ba0ad46a3ee8e9e9266e9。26工具身份含9个真实sanitizer runtime payload，82个worker raw/effective/canonical环境有据可查；每次编译前即选择GPU1并验证实际context UUID，加载后再次核对。GPU0外部任务未停止或用于本任务。

首轮run-current-gpu-closure-20261102T075639Z完整保留failed：22生成成功，首diamond普通GPU数值通过，但memcheck在首instrumented API前因合法LD_LIBRARY_PATH注入被严格环境守门拒绝，不能计作安全通过。新fresh运行只允许named+load-only安全worker使用与冻结实际工具目录和9payload绑定的精确前缀，后接原始LD原串；未知/额外/换序路径仍拒绝，raw/canonical双记录。原source前像、CPU探针、20项driver测试及三文件bridge保留；fixture只新增protocol.safety.environment说明，shape/输入/reference/阈值/backend/CPP/预算不变。原运行没有被修改或并入接受样本。

最终CPU：635项原生单元（72 suites）、45项选择测试、CTest10/10；执行Python19脚本共186项通过（含新support12、driver20、demand5及export4）。首Python汇总中test_export漏传OPT参数导致入口错误，原日志保留，正确参数后4项全过；最终summary逐脚本选择可追溯的有效结果，不覆盖错误调用。harness环境修订后再次CTest10/10、export4、driver20和native export9全过。

数据独立审计 ../optimize_plan/current_gpu_closure_artifacts/review/final-data-audit.json（SHA53b98a399dfbb0b8a1c84d7781119b00ac2577319c53fa1e10deaed347f290ea）；身份/资源独立审计 ../optimize_plan/current_gpu_closure_artifacts/review/final-identity-resource-audit.json（SHA8c17c1009148bf3a55bdda7ba9378c68e7520a7d7d51bd43422153a9636361c2）。源码层审查0 open Critical/Important；最终whole-stage审查与统一索引共同绑定本报告及全部原始证据。

## 明确保留的边界与后续

本轮的闭包声明限于预登记22自然配置；更广数学合法布局、推断可评分或checked emission不自动有GPU资格。保留logical footprint65536、256-bit代数、非dyadic、Copy/Reduce/MMA及大输出codegen等边界；新大源路径只支持Register和受限纯f32 DAG。任意大stride/view的u32 root-relative地址可表达性、任意控制流、动态shape、multi-CTA大算子分块、尾部mask、流水重叠、warp specialization/cluster/TMA输出Store不在本阶段范围。

当前支持域闭环完成后，下一阶段仍需在新数据上验证默认模型与选择质量、充分知情基线及公平比较协议。本报告不宣称纯静态物理寄存器预测达标、模型外推可靠、J优于LI或优于Triton/TileLang。

## 可复放交付与旧证据保护

source-final-manifest.json归档394文件；14文件源码补丁（8修改+6新增）已git apply --check、apply并逐字节roundtrip。五个旧stage共22031封存条目、5449285586字节及5个index自身SHA保持，9个前像匹配上一symbolic快照（cost_calibration.py未改）。原Git HEAD不变、暂存区保持空，本阶段未提交或推送。完整计划、原始运行、源码/工具、支持矩阵、独立审查与统一evidence-index.json均在本阶段artifacts；索引不包含自身哈希。
