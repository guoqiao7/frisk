# P1.3c：静态循环频率实现与验收报告

日期：2026-10-29。状态：P1.3c已完成验收。实现、功能回归、整体源码/驱动复审、正式观测及最终数据/文档独立复审均通过。本报告交付范围为P1.3c；单位轴支持、GPU后端、消融、校准、搜索优化和跨框架比较属于后续独立任务。

## 实现结果

新增显式可选的静态循环次数模型，贯通目标、求解、转换/重计算选择、计划复验、物化、转换后优化及诊断。默认仍为legacy。

```sh
frisk-opt input.mlir --frisk-layout-pipeline='boundary-frequency=static-trip-count'
```

`frisk-infer-layouts`和`frisk-optimize-layout-conversions`也接受同名选项；合法值仅`legacy`与`static-trip-count`。非法值在IR修改前失败。

| 模型 | 完整模型ID | 循环计费 |
| --- | --- | --- |
| 默认/显式legacy | `sm90-layout-lex9-legacy-v1` | 保留原`scf.for/scf.while`深度的`2^depth`及历史饱和行为 |
| 显式static-trip-count | `sm90-layout-lex9-static-trip-count-v1` | 对可解码的常量有符号`scf.for`正step计算次数；其他循环使用明确标记的factor-2代理 |

频率只乘`conversionBytesAndSync`中的转换/clone动态边界工作。代码大小、静态资源、普通算术诊断、候选、硬契约、外部ABI和默认求解预算不变。本阶段没有将活跃峰值写入成本目标，也没有建立完整指令时延模型。

## 频率语义与边界

- 次数为`max(0,ceil((ub-lb)/step))`，使用扩宽APInt计算，避免有符号差值和向上取整溢出；宽常量不截断为64位。
- 嵌套循环相乘。任何已知零次祖先优先得到精确零，并清除频率自身的proxy/saturation标记；已有其他成本的饱和标记不会因此被抹掉。
- 恰好等于`UINT64_MAX`可表示；更大乘积饱和。零动态工作乘饱和频率仍为零，不凭空产生新的动态溢出。
- 动态边界、unsigned比较、while以及不能可靠解码的LoopLike循环使用factor=2，标记proxy。该代理不是次数上界。Legacy只计原先的for/while，不因helper识别更多LoopLike而改变旧行为。
- 频率描述外层循环次数的模型，不建模分支概率，也不等同于实际机器指令数、硬件传输量或运行时间。

静态模型中的受限外提要求严格动态收益：零次和一次循环保持转换原位置；100次的合法register-only转换可外提为一次。既有支配、循环不变、硬属性、region和副作用约束继续生效；iter_arg、shuffle、Shared通信和屏障等安全边界未放宽。

## 一致性与可核验诊断

每个target实例持有频率模式；转换边、clone边界和optimizer使用同一模式。物化API新增末尾可选借用target参数，infer传入实际target；旧调用保留legacy回退。新增边元数据参与计划相等比较和重新计算验证，不用模型ID字符串猜测配置。

原`conversionBytes`保持未加权语义和既有聚合行为。新增`weightedConversionBytes`及独立proxy/saturation信息，按唯一Convert leader计费，不混入clone、同步或代码大小。旧raw字段的溢出行为没有在本阶段改写。

`finalCost`和选择统计仍属于转换后优化之前的选中方案；它们不是对hoist之后IR重新评分的结果。实际IR的算术/资源诊断仍单独发布。人类可读的coverage文本已按模式区分，legacy文本保持原样。

## 可复核的100次循环见证

独立fixture固定两种register布局，在100次循环中插入一个必要转换：

| 项目 | legacy | static-trip-count |
| --- | ---: | ---: |
| 选中方案的频率因子 | 2 | 100 |
| 动态转换成本，第4项 | 1024 | 51200 |
| 其他八项 | `[0,0,0,28,0,0,1,0]` | 相同 |

这验证真实次数替换深度代理，不能解释为50倍加速。该fixture的后优化会把合法转换外提，所选方案的上述成本仍保留原阶段含义。0/1/100次外提位置另由独立单测验证。

## 验证与复审

Task1公共helper：13项Debug与13项Release通过，先记录13项运行中8项行为失败的RED。Task2集成：Release定向124项、Debug定向125项通过；后续元数据25项和coverage文本1项在两个配置分别复测。上述运行有重叠，不相加冒称独立测试总数。Task2新增8个测试身份在两个配置均覆盖，另有3个相关lit在两个配置通过。

覆盖包括零/一次/百次、负下界、非整除step、嵌套已知/未知、零优先、宽整数及饱和、静态外提安全、多消费者去重、两种模型的有限域oracle/partial bound/refinement、跨模式计划元数据拒绝、非法CLI、早期失败和回滚、自定义target身份独立性。

独立Task1复审通过；独立Task2复审通过。后者发现coverage文字仍沿用legacy，已以断言RED→Debug/Release各1项GREEN修复并复审关闭。

最终功能回归结果：

| 验证 | 结果 |
| --- | --- |
| 完整Release主单测 | 570/570，66个suite；无失败、错误、disabled或skip |
| Release选择质量单测 | 20/20，1个suite |
| 新增测试身份核对 | 相对原549项新增21项，全部包含于Task2 Debug最终/late结果；无旧测试被删除 |
| 不同lit用例 | Debug 9/9，Release 12/12，含Task2各3项与根控制者额外6/9项 |
| CTest守门 | 5/5，包括选择质量Python runner |
| normalization/API/SM90工具 | normalization 5项通过，API及SM90守门退出0 |

## 正式语义核验与CPU观测

整体源码和测量驱动复审批准后，先完成18次语义调用，再进行36次串行CPU调用（6次预热、30次测量）。三类输入各比较旧工具、新默认、新显式legacy、新static，并分别对两个显式模式的结果重放。

- 所有输入的旧工具、新默认与新legacy保持IR、完整九项成本及既有非计时统计字段一致。
- 无循环的两类Attention在static下也保持相同IR、完整成本和既有非计时统计字段；新模型ID和新增模式元数据按预期不同。
- 百次循环的动态项精确为1024/51200；原始转换字节512，加权字节分别1024/51200，唯一转换数1，proxy/saturation标记与模型一致。两种模型在这个百次循环上的最终IR也相同。
- 重放核对IR和实际IR诊断，不要求重放后重新构建的求解统计与首轮完全相同。
- 结束时重新校验工具、驱动、输入（含仓库外loop fixture），并另外核对全部329文件的最终源码清单；身份均未变化。

每个配置先预热1次，再取5个完整CLI墙钟时间样本的中位数：

| 输入 | legacy中位数（秒） | static中位数（秒） | static / legacy |
| --- | ---: | ---: | ---: |
| 百次循环小图 | 0.015832366 | 0.016886756 | 1.066597 |
| 自由布局Attention | 13.154815970 | 13.165764287 | 1.000832 |
| 固定ABI Attention | 7.381510777 | 7.385568639 | 1.000550 |

小图观测增加约1.054毫秒（6.66%）；两类Attention中位数差异小于0.1%。没有CPU隔离或随机化顺序，不能据此作显著性判断。测量包含CLI启动、推断、诊断与输出，不是GPU时间，也不能据此宣称推断已加速。原始命令、每次样本和RSS均保留在[测量结果](../optimize_plan/p1_3c_measurement_results.json)。

## 过程记录与限制

- 保留真实测试时序：Task2在生产改动前记录新CLI缺选项的RED，外提分支修改前记录一次循环错误外提的行为RED；数值旧工具对照及部分其他回归并非逐个先于实现运行。最终GREEN不能被描述为严格逐测试先红流程的证明。
- 一次溢出测试误将`2^63`当作溢出；修正fixture到`2^65`，未改生产代码，初次失败日志保留。
- 构建记录包含既有CMP0116/Ninja恢复提示，以及GCC对已有一位APInt构造的告警。相关helper未改变，告警没有被隐藏；不把构建日志称为完全无告警。后续coverage断言修改了SM90测试文件，因此早期“整文件与基线相同”的审计结论只适用于该修复之前。
- 本轮没有GPU执行或性能领先证据。静态模型仍是有限覆盖的代理，动态循环factor-2和未建模的latency/cache/overlap等边界继续存在。

## 来源、证据与下一任务

源码基线是`/tmp/frisk-p13c-20261029-wrrdckp5/tree`的324文件实际工作区快照，包含此前未提交实现；不是仅Git HEAD。Task2基线327文件，当前源码清单329文件。旧工具另行保存且仅链接系统动态库，不会因新项目库重建而改变基线逻辑。

本轮直接修改原目录，保留已有工作，不stage/commit/push或创建worktree。原始证据目录为`/tmp/frisk-p13c-20261029-wrrdckp5`，持久证据见[归档索引](../optimize_plan/p1_3c_artifacts/README.md)，阶段及总路线见[执行计划](../optimize_plan/p1_3c_execution_plan.md)。源码增量共36个文件（31个修改、5个新增），最终清单329文件。

下一独立任务为最小形状支持：非零rank的静态正二次幂形状允许单位轴，补齐插入轴Broadcast、rank≥2到rank≥1 Reduce和必要静态view，再单独验收。它不由P1.3c完成自动获得。
