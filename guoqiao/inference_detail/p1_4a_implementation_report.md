# P1.4a 最小形状支持实施与验收报告

日期：2026-10-29。状态：P1.4a已完整验收。实现、任务复审、整体源码及最终证据/文档独立复审全部通过，无未结问题。

## 范围与实现

静态正二次幂形状允许extent=1，单位轴贡献零逻辑位。支持非零rank tensor的ownership、插入轴Broadcast、rank≥2到rank≥1 Reduce，以及必要的静态tensor.expand_shape/collapse_shape和transpose。Broadcast保留排序互异插入轴与同线程可用性要求；Reduce保留规范contributors与执行契约。

静态view依现有TransformLayout关系实现row-major元素身份的bit行重排，正反传播、物化类型和actual-only验证共用该语义。没有扩大为任意reshape算法。

rank0、动态、零extent、ragged/non-power-of-two仍在功能入口拒绝；不改变ABI、目标、搜索预算、模型默认值或GPU后端。底层既有proof helper的独立rank0历史行为未在本阶段另行改变。

## 测试与独立审查

初始行为RED共6项运行中4个新增测试按预期失败。首次GREEN暴露LayoutCostModel::logicalSlot对尾部单位轴extractBits(width=0)的断言；已令零宽坐标为零。静态expand_shape缺layout model也由真实失败发现并实现。过期测试将单位轴判unsupported的期望按新契约调整，并保留zero等负例。

独立任务审查发现ownership/accessor仍进入零维GF2 factory，修复显式前置guard并保留replication/writer检查。整体审查进一步发现shared attribute getMatrixValue同类入口，已在APInt构造和GF2调用之前拒绝空维度并获独立复审批准。内部guard维持既有可观察failure语义，未编造新增行为RED。

最终验证结果（重叠运行不相加）：

| 验证 | 实际结果 |
| --- | --- |
| 完整最终Release主单测，唯一一次 | 575/575，66 suites，261.510秒；unique575，0失败/错误/disabled/skip |
| 最终Release选择质量单测 | 20/20，172.778秒，最终源码重链接 |
| 受影响单测（最后attribute内部guard之前） | Debug112/112，577.091秒；Release112/112，62.101秒 |
| 最后内部guard最终源码小聚焦 | Debug15/15，Release15/15，包含全部5个新增身份与零bit/成本覆盖 |
| 相关lit | Debug5/5、Release5/5，最后guard后再次确认 |
| CTest守门 | 最终5/5；selection Python25项包含于其中 |
| Python/工具 | normalization5/5，API及SM90工具退出0 |

5个新增测试身份均同时包含于最终Debug15与完整Release575，详见artifacts/new-test-coverage.json。没有删除旧测试；单位轴旧拒绝期望更新为合法，保留非法零extent负例。

任务复审修复后源码批准；最强可用gpt-6-astra整体源码审查及窄修复复审批准，无未结Critical/Important/Minor。任务和整体报告分别为artifacts/task-1-review.md、whole-review.md。全部336源码SHA及18组基线/最终增量哈希经整体复审重算一致；最终控制者再次核验无漂移，所有已知任务/控制者jobs已关闭。

## 过程与边界

Debug Ninja出现premature end of file恢复提示，多次触发139步重建；原因未定位，构建成功后继续验证。预任务review修复的旧Debug run被定点取消RC-15，日志保留，不计为通过。llvm-lit直接执行shebang因权限失败，改用python3启动同工具后通过；保留原始失败和更正命令。

直接修改H100 /home/baopeihua/frisk并保留已有dirty代码；未worktree/stage/commit/push，无GPU执行及性能收益结论。新实际基线/tmp/frisk-p14a-20261029-57dtdpio/tree共335文件，区别于HEAD；最终源码新增CLI fixture后336文件。

执行计划：../optimize_plan/p1_4a_execution_plan.md。持久证据索引：[p1_4a_artifacts/README.md](../optimize_plan/p1_4a_artifacts/README.md)。


最终守门过程另记录两项setup修正：selection单测重链接后，CTest使用的独立selection benchmark尚未重链接，首轮5项中该一项因source manifest mismatch失败（Python25中2项）；补建benchmark后仅重跑失败CTest通过，不修改生产源码。API首轮漏必需--source-root，参数退出2；更正后通过。原始日志及更正命令保留，不将失败隐藏为首次全绿。

本次仅交付P1.4a形状闭环。下一独立任务是GPU基础后端，后续异步路径、消融、成本/资源校准、搜索开销优化和公平跨框架比较仍未完成。

最终证据审查确认两源码包335/336成员哈希、18组delta及全部持久文件校验有效。审查发现完整上下文review diff漏最后guard而短diff完整，已从实际基线重生成803行最终diff并复核关闭；不涉及源码变化或测试重跑。
