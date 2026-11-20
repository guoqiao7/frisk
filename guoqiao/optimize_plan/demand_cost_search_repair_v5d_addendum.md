## 2026-11-03：需求覆盖与同预算搜索复验（整体未通过）

本轮执行覆盖与预算内搜索修复已经实现并复验：114配置、16002数值/保护记录、144次实际sanitizer全部通过；36次受控Oracle和预算核对通过。原始计时150个配对批次/1800事件，无无效或替换批次。

整体模型及选择质量仍未通过：组合主时间比0.940789，但exp32×32退化44.9%；固定新搜索时，新成本模型反慢6.38%。自然域regret和单例保护失败，默认保持legacy-v2，不进入公平对比版本冻结。收益主要来自搜索修复，不能称新模型已经取得性能优势。新模型CPU solve合计约为legacy的9.25倍，仍需优化。

完整报告：`guoqiao/inference_detail/demand_cost_search_repair_report.md`；证据与保留的CPU失败历史在`guoqiao/optimize_plan/demand_cost_search_repair_artifacts/`。新性能标签只在冻结后的R4读取，没有调权或换case；以下原文逐字节保留，作为历史状态。

---

# v5d预留出模型候选变更

17特征v4d/r2仅为已知训练候选。全34镜像的独立排序诊断确认受控最优6/6，自然最优2/6；mixed存在17特征完全相同、时间显著不同的镜像。r2原profile、注册、报告及字节桥接证据全部保留，添加SUPERSEDED旁证，没有运行六个新留出的GPU时间。

新候选只增加第18个通用结构特征global_load_issue_groups；profile model为sm90-layout-demand-calibrated-v2，语义ID为v5d，保留旧v1/v4d的17特征接口。每4个ld.global.u8构成一个f32请求；浮点mul/sub/div/ex2、shuffle、shared load/store、barrier和global store截断连续load run；地址生成、字节组装、literal moves和Register no-op不截断。不增加第19个affine-basis特征，不改变kernel发射。

C++按实际调度的GF(2) affine-query顺序计数，独立Python审计从后端实际PTX行顺序重建。NNLS方法、全部34个已知训练镜像和固定aggregate-bank0均保留；训练秩亏、inactive一tick先验和已知排序都需明确报告。新候选必须重做完整标签桥接和冻结后验收。114配置、同算法/ABI/精度/候选域、六个未计时的新N留出、三个已观察Attention锚点及原质量门槛均不变。已知排序改善不等于留出通过。
