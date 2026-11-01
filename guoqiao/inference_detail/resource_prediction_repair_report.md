# 资源预测深入归因与修复报告

日期：2026-10-31。状态：实现与66候选数据独立审查通过；通用无编译物理寄存器点预测仍未通过资格。仅服务器 `/home/baopeihua/frisk` 修改，未暂存、提交、推送或保留本地代码副本。

## 结论与修复口径

本阶段修复默认资源报告的结构性错误和可信来源。Shared改为native编译前的结构分配计划，计入对齐、别名root、串行复用scratch和TMA barriers。寄存器撤下缺乏证据的SSA→physical点值；PTX输出仅为typed虚拟压力，物理数量标记requires_compilation，再由精确cubin的编译器资源反馈与实际loaded image/entry的driver信息确认。compiler feedback的零差错不能称为编译前预测MAE=0，也不消除首次native编译成本。

C++搜索期原有加性logical-slot风险proxy仍保持代理语义，未宣称整个搜索阶段的寄存器估计都已精确。无需编译的通用physical register预测本阶段不发布，coverage=0、MAE=None、not-qualified。

## 已查明的原因

**Shared输入遗漏与不可识别回归。** 上轮两个root特征在数据中高度共线(rank2/3)；control图两项均为0，实际转换却需要0/4096/8192B scratch，因此拟合成约1450.67B截距。此前的误差并非只靠调整系数可解决。

**寄存器输入与分配阶段不匹配。** 当前适配器将整个kernel装入一次opaque inline PTX；高层SSA tensor存活未覆盖完整的地址、owner、字节组装、转换和IEEE除法展开及ptxas调度/分配。所有声明虚拟寄存器也不能当物理数：mixed A/B声明总GPR达1375/3055，typed虚拟live峰值68/85，实际分配80/168；remat A/B虚压71/69却分配80/96，连简单的大小顺序都不能保证。

固定同一PTX与实际bundled ptxas12.8.93，仅改变优化等级的CPU对照：

| case | O0 | O1 | O2 | O3 |
|---|---:|---:|---:|---:|
| mixed64×32 A | 83 | 78 | 80 | 80 |
| mixed64×32 B | 91 | 115 | 168 | 168 |
| remat128×16 A | 77 | 92 | 80 | 80 |
| remat128×16 B | 75 | 77 | 96 | 96 |

16次编译日志明确stack/spill stores/spill loads均0；不能把原误差归因于spill。原mixed A/B物理live峰值78/165、remat A/B 78/94，贴近80/168/80/96的allocated数量。峰值区域包含弱对齐LDG.U8、地址IMAD及跨转换保留。仅CPU的div→mul与byte load同步扰动说明除法展开、地址/搬运和调度共同作用；这些改变算法/调度的探针不作为正确性或性能收益证据。

**工具口径差异。** 旧103产物中38份raw/user Shared均0；65份非零cuobjdump raw SHARED比driver用户容量多1024B，匹配这台H100的device RESERVED_SHARED_PER_BLOCK。例如mixed B raw9216、user8192，TMA raw7192、user6168。不能对零值也减1024或把raw误称用户分配。新报告分列raw/user/reserved，并根据明确设备属性和driver确认。early local-array256探针显示STACK256、rawLOCAL0、driver local256，故rawLOCAL0也不能当local0或spill计数。

## 实现

- `shared_allocation.py`：受检export与实际redistribution route构造分配计划；独立区分root footprint/padding、串行max scratch、TMA barrier与对齐，预算/溢出/未知重叠或控制域明确拒绝。emitter复用；103旧TTGIR与原config字段逐字节不变，结构plan=独立packing oracle=历史driver user Shared。
- `register_pressure.py`：类型化PTX CFG、64位双word、predicated写、向量tuple、DCE和地址依赖；reverse worklist与全局word-work预算，超限/未知语义返回unavailable。54旧PTX与预算修复前features/状态一致。仅提示虚拟压力，physical始终None。
- `resource_prediction.py`：区分pre-native plan、PTX-only信息估计与compiler-exact反馈。schema2缓存绑定实际cubin/tool/entry；查询owned字节快照并在前后及cache-hit返回检查原image/tool，损坏/变化failclosed。raw Shared不冒称用户容量，nonzero STACK/LOCAL保留driver确认。
- 正常`cost_calibration.compile_exact/execute`先写Shared计划、再写仅PTX输入记录（combined Triton.compile之后、首次resource feedback之前），然后查询native资源并绑定实际loaded binary SHA/entry确认driver。不得把第二次查询时间冒称首次反馈前时间。
- 默认`calibration_analysis.resource_calibration`转为上述来源；缺lowering/native证据明确unavailable；历史SSA回归仅显式legacy且unqualified，不再默认发布错误physical点值。

API与原因依据参见[NVIDIA CUDA 12.8 binary utilities](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-binary-utilities/index.html)和[PTX寄存器定义](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)。PTX分析与最终分配分层，不从虚拟寄存器声明推断机器分配。

## 新验证与结果

原12spec（包含上一轮3个留出）均只作known；新6spec和A/B×C/O共24prospective在任何新native结果前登记：diamond16×64、remat32×64、mixed128×32、div6 64×64、conflict256×8、exp1 16×128。注册SHA `b02e0ada1cf04079c67aa3012b1a45319f82c613190c1be1c241b83eba79d6eb` 未改变，不按结果换形状、调阈值或拟合。

最终接受run为 `resource_prediction_repair_artifacts/run-resource-repair-20261031T150743Z`；known42后冻结null physical模型，再验证prospective24，不做kernel performance timing。

最终66个完整配置全部验证通过，其中42个known、24个新prospective；24个新候选未参与任何寄存器模型拟合。独立数据复核见 `resource_prediction_repair_artifacts/review/final-data-check.json`。

| 验收项目 | 全矩阵结果 | 解释 |
|---|---|---|
| Shared执行级编译前结构预测 | 66/66误差0B，新24/24误差0B | 输入为execution/route，不读native或driver真值 |
| 编译器寄存器反馈与driver确认 | 66/66一致 | 编译后反馈准确性，不是预测MAE |
| raw Shared/user Shared/device-reserved | 66/66关系一致 | 30份双零，其余按设备reserve=1024B核对 |
| PTX typed virtual pressure | 66/66 available | 仅虚拟压力提示，不是物理点值或上界 |
| 纯静态physical register点预测 | 新候选覆盖0/24，MAE/max-under=None，not-qualified | 未发布无依据点模型，首次仍需编译 |
| 普通/stream/capture数值与护栏 | 4674条通过 | 包含输入不变与guard记录，不是4674个独立kernel |
| 四种sanitizer | 54配置×4=216次实际通过；12配置精确复用 | 复用要求raw cubin、完整config、physical ABI全相同 |

新形状进一步验证了分层的必要性：mixed128×32有PTX虚压132/165而实际168/255；div64×64虚压71/101而实际80/128；conflict256×8虚压52/69/71而实际32/96。不能用虚压做物理寄存器上界或直接作occupancy容量。本矩阵driver local均0，不扩张为对未知spill/非零local域的验证。

独立审查重新哈希2004份接受产物、371份测量期源文件和11项工具身份，确认登记、known冻结、null模型冻结均早于prospective输入；完整报告和最终源码身份另列，不能回写测量期manifest来掩盖后续测试变化。

## 回归、历史与交付

独立源码审查0Critical/0未关闭Important；定向含fixture93项通过；既有与新Python兼容回归136项通过；更新source manifest/native工具后CTest10/10通过。preflight局部补充15测试和独立复核通过。额外显式入口export3/3通过；backend原6项中1项测试夹具因根容量未扩大而提前拒绝，保存失败日志后只修fixture，6/6通过。C++推断逻辑未改变，生产Python在66矩阵后也未改。

后续唯一源码差异为 `benchmark/layout/execution/test_backend.py`：f16→f32拒绝测试现在同时扩大相同root的所有alias和canonical root地址，保持容量合法，继续断言Copy位宽合同拒绝。生产实现、登记spec、实际输入、工具和接受产物不变。旧测试前像与SHA、失败与通过日志、measured/final身份映射独立保存；测量manifest不回写。严格source guard在这个测试hash变化后会拒绝原run继续重跑，这是预期保护，后续运行应另freeze新源身份。

保留首次run的空GPU进程查询预检失败：42 CPU-generated、0 worker、0validated、0prospective，原manifest不回写冒称failed，另有failure-accounting。修正成功empty stdout视为idle，UUID/CSV/退出码/stderr/foreign仍failclosed；fresh run同注册重新执行。early schema1 local语义探针只作诊断前像，不计入新schema2验收。

前一阶段sealed scope的4751项已独立逐字节/SHA核对，旧index自身及报告/计划保持原SHA；本阶段合法更新的current源码和长期文档不是旧sealed目录，不能混称prior index全部current路径仍未变化。

交付入口为本报告与 `resource_prediction_repair_artifacts/evidence-index.json`。统一实施计划记录协议。测量期source identity按后缀规则包含371份文件；源码包按更广范围扫描包含380份（含README等），原365份起点另补5份与HEAD逐字节一致、初始枚举遗漏的输入；差异实际为7修改+10新增。`source-final-manifest.json`、`stage-delta-files.json`、`stage-implementation.patch`和`source-package-audit.json`提供370份基线（365原前像+5份未改HEAD补证）到380份最终源码的可复放差异；`measured-final-source-bridge.json`明确测量/最终测试差异；`long-term-document-update-audit.json`核对5长期文档原字节后缀。独立整体意见在 `whole-review.md`。源码和证据仅服务器保存，不自动提交。

## 后续使用边界

正常`cost_calibration.compile_exact/execute`已接入Shared结构计划、typed压力诊断、cubin缓存与实际loaded image确认；默认资源分析不再静默输出错误SSA物理回归。其他调用方在使用资源做occupancy或硬容量筛选时，应读取compiler/driver确认字段；PTX提示不可替代它。未支持的控制/重叠/未知PTX和预算耗尽明确unavailable/拒绝，不猜资源。当前C++搜索期logical-slot代理和尚未改善的时间模型仍有局限；本阶段未宣称提高搜索决策质量、kernel加速或相对Triton/TileLang优势。
