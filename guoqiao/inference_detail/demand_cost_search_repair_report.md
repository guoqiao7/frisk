# 需求覆盖、同预算搜索与新性能留出复验报告

日期：2026-11-03（Asia/Shanghai）。仅在 H100_direct 的 `/home/baopeihua/frisk` 修改和留证，没有本地代码副本。

**结论：功能与执行账本、预算内搜索修复通过；整体模型及选择质量验收仍未通过，不进入公平对比版本冻结，也不晋升新模型为默认。**

新组合的六例几何平均时间比为 0.940789，95%CI [0.938762, 0.943415]，约 1.063× 加速；但 exp32×32 退化44.9%，违反单例保护和regret门槛。固定新搜索后，新成本模型比legacy慢6.38%；合并收益主要来自搜索修复，不能归功于新成本模型。

## 验收证据

| 项目 | 结果 |
|---|---|
| 全量构建 / CTest / lit | 通过；11组CTest，80项lit |
| C++布局 / 选择与Oracle单元 | 670 / 48项通过 |
| 实际GPU配置 | 114/114通过，GPU1 UUID固定 |
| 数值、保护、stream/graph | 16,002条记录、3,906个输入/参考/偏移组通过 |
| 四工具安全检查 | 144个实际进程、480条子集记录；78配置精确复用 |
| 执行覆盖 | 96纯DAG exact，其中48个新模型18维计价对账；18个Attention明确fallback |
| 有限域Oracle / 预算 | 36/36通过；最高9947 probes、248 refinement，未超过65536/256 |
| 原始计时 | 15组、150配对批次、360 image-block、1800 events；0 invalid / 0 replacement |
| 独立复核 | 原始文件/事件核验通过，300项统计数值与官方分析一致 |

## 冻结规格与方法

模型为legacy-v2与18维v5d；每种模型分别比较充分知情LI、J-old(seed off)、J-new(seed on)。J-old指同一修复源码中的搜索消融，不能称旧二进制版本。受控域再含强制A/B，共六例×(10受控+6自然)+三层Attention×6=114配置。候选域、算法、ABI、精度、beam32、exact256、总probe65536及refinement256相同。不同臂实际使用的probe数不必相等。

| 新性能留出规格 | 形状 | N |
|---|---|---|
| diamond | 128×32 | 4096 |
| remat | 64×64 | 4096 |
| conflict | 128×32 | 4096 |
| mixed | 128×64 | 8192 |
| div6 | 64×16 | 1024 |
| exp1 | 32×32 | 1024 |

六例的完整规格及同case_id/tail/ABI/dtype/threads下的N均不同于此前29个已登记身份。其中diamond/remat曾在R3做CPU预检并用于修复最终store端点遗漏；六例GPU性能标签均在R4首次读取。CPU历史及失败原样保留，没有换case或放宽门槛。此后六例均属已观察数据，不能再次称新留出。Attention weak-vector/strong-cpasync/strong-tma的64×16仅为已观察回归锚点，不混入主六例收益。

训练固定使用此前34个已测精确镜像，来自9个完整规格（含三个Attention ABI锚点），全部保留；新18特征通过完整赋值重放提取，TTGIR/config/cubin与原标签镜像字节桥接。R3→R4仅修复通用计费缺口，profile与fit字节完全相同，未用新性能调权。

每组10个配对批次，每镜像每批5个event；按完整执行身份去重，别名仍保存。图节点15–1437，均不超过4096；输入/参考/偏移在批内配对。计时为普通缓存条件的kernel graph事件，编译、加载、描述符、捕获与搜索成本分别报告。

门槛不变：regret几何平均上界≤1.03、最坏上界≤1.10；主时间比上界<1/1.03；九个自然案例各自以alpha=.05/9检验退化上界≤1.05。受控参考是精确两赋值；自然参考是六策略所选镜像并集，不是整个原始候选域的GPU最优。

## 结果与归因

| 比较（时间比，<1较快） | 比值 | 95%CI |
|---|---:|---|
| 组合：demand J-new / legacy J-old | 0.940789 | [0.938762, 0.943415] |
| 固定新搜索：demand / legacy | 1.063797 | [1.062950, 1.065437] |
| 固定旧搜索：demand / legacy | 1.124862 | [1.119735, 1.127840] |
| 固定demand：新搜索 / 旧搜索 | 0.836914 | [0.834138, 0.840814] |
| 固定legacy：新搜索 / 旧搜索 | 0.884369 | [0.882556, 0.886471] |
| demand J-new / 充分知情demand LI | 0.911508 | [0.909322, 0.914829] |

所有受控策略均qualified。新模型J-new受控regret=1；充分知情LI受控最坏约1.0094，也通过。自然域六策略全部failed：

| 自然策略 | regret几何平均 [95%CI] | 最坏 [95%CI] |
|---|---|---|
| demand-J-new | 1.084914 [1.083447, 1.087299] | 1.449282 [1.442369, 1.462742] |
| demand-J-old | 1.296327 [1.290814, 1.300954] | 2.142526 [2.114073, 2.156918] |
| demand-LI | 1.190242 [1.186216, 1.193605] | 1.614251 [1.586715, 1.633654] |
| legacy-J-new | 1.019850 [1.018712, 1.021237] | 1.125171 [1.117658, 1.134381] |
| legacy-J-old | 1.152708 [1.150737, 1.155428] | 1.649827 [1.645038, 1.671380] |
| legacy-LI | 2.514290 [2.500400, 2.520175] | 29.683495 [29.503720, 29.927866] |

逐例时间（µs）与新模型J-new regret如下。时间是batch中位数的聚合；regret按配对比值计算，不能直接用表中聚合时间相除重构。

| 自然案例 | demand J-new | legacy J-new | demand LI | 已测最快 | demand J-new regret |
|---|---:|---:|---:|---:|---:|
| holdout-diamond-128x32 | 6.1029 | 6.1029 | 9.5651 | 6.1029 | 1.000000 |
| holdout-remat-64x64 | 6.1153 | 6.1153 | 9.8205 | 6.1153 | 1.000000 |
| holdout-conflict-128x32 | 6.1584 | 6.1584 | 6.1584 | 5.4563 | 1.125171 |
| holdout-mixed-128x64 | 10.8800 | 10.8800 | 10.8800 | 10.8800 | 1.000000 |
| holdout-div-64x16 | 3.8926 | 3.8926 | 3.8926 | 3.8926 | 1.000000 |
| holdout-exp-32x32 | 3.4075 | 2.3403 | 2.3403 | 2.3403 | 1.449282 |

exp的九案例familywise区间为[1.437828,1.470523]，明确超出1.05。Attention各策略为相同镜像，时间比均1，不能宣称独立联合选择收益。

冻结并集交叉计分确认：J-new在六个新自然规格中均已达到所测并集内最小v5d完整tuple，并且不高于同模型LI/J-old；没有相对这些实际证书的更低模型成本遗漏。conflict与exp是明确的模型倒序，而非这里已知的搜索缺口。不据此声称自然域全局最优或对任意LI都有普遍保证。

## 剩余模型缺口

exp：模型选择1,599,638 ticks的Shared方案，而更快的无Shared方案为1,798,427 ticks。两者有效Q/K地址族相同、GPR32/LOCAL0；慢方案有3次Shared交换、24KiB转换、6barrier。其ALU/Shared bytes/barrier权重均为0，不能解释为这些工作物理免费。

独立native诊断还确认：快的eager方案PTX中96条byte LDG有32条unused-V读被ptxas删除，native只剩64；慢方案PTX/native均64。因此发射层计量一致不等于编译后存活工作已经一致。该例主SASS慢1158/快458，额外24STS+24LDS+6BAR。

conflict：新搜索选中的模型成本6,557,243低于更快J-old的6,759,776，却慢12.5%。两边native load/math数量和有效地址族相同；emitter的load-group1对32并未代表native调度差距，首个math前native LDG为29对38。不能把此失败无证据归于小N阈值或寄存器阈值。

仅调load-group系数不能同时修复两例（固定其余权重）：exp需w>257728，conflict需w<52405.677。单独剔除已证unused-V读收费会反转exp这一对，但会加强conflict的错误偏好，不能称统一DCE修复已经解决二者。下一轮应分别对齐编译后存活工作/调度证据，并独立校准通信边际费用；不应继续扩大beam或在本留出上反复调权后宣称通过。

## 校准精度与CPU代价

训练NNLS RMSE 0.656020µs、设计rank14/16（含intercept），仍秩亏。12个非单例已知组的top选择全部正确，但自然21个可区分pair仍有1个非最优镜像倒序；训练top正确不等于完整排序或泛化合格。intercept 1.231476µs只用于预测诊断，不参与同程序选择。

按预注册仅统计demand所选赋值的26个新性能留出group-unique镜像（受控12/自然14），时间代理MAE为8.2183µs，受控15.7126µs，自然1.7946µs；已观察Attention锚点0.4643µs，不能把锚点精度当外推。两个训练全0特征(register/conservative bytes)在本轮57个demand配置均未激活，其1tick系数仍是先验，不是校准事实；有训练激活但NNLS得到0的ALU/Shared bytes/barrier，也没有独立可辨识性证明。

六新自然规格的CPU合计如下，为单次描述性观测：

| J-new | 内部solve计时 | 原生进程端到端 | probes |
|---|---:|---:|---:|
| legacy | 5.2911s | 15.4946s | 19964 |
| demand | 48.9323s | 59.1686s | 19982 |

新模型内部solve约9.25×开销，probe数几乎相同。端到端还含恢复域/证明、物化、导出与诊断，不应称纯搜索时间；没有独立seed墙钟。模型语义稳定后仍需缓存完整赋值的解析特征/证明及减少重复需求构造，新模型的CPU开销仍明显，需要另行优化；本轮没有预注册CPU墙钟晋升门槛。

物理资源范围：GPR32–255、用户Shared0–32768B、driver LOCAL0–21912B/线程。15配置/4镜像存在非零LOCAL；spill次数未知。尤其mixed legacy-LI低GPR32却有21912B LOCAL，不能用低GPR或理论高residency解释性能。

## 实现与根因

旧访问成本按整个硬件载体计价，而大复制布局的后端按消费者需求发射。已知 mixed 的旧访问费用为4192个逻辑warp访问，实际为128，相差32.75倍；修复后按实际24个load query和8个store向量核对，避免将逻辑复制量直接当执行工作量。

Frisk把每个需求表示为 $q(t)=c\oplus\bigoplus_{i=0}^{6}t_i a_i$，即128线程到寄存器索引的8项GF(2)仿射映射。Register转换通过同线程寄存器子空间的最小规范截面组合此查询；缓存键为SSA与完整仿射查询，保留不同查询的实际执行重数。有界Shared/Shuffle边界完整物化原source与destination，不把通信悄悄替换成重计算。地址、first-owner、bank读写方向和同步均与后端独立核对。

17项总量特征仍不足以刻画发射结构。四组已知反例的“慢−快”差均为access −64、sectors −320、ALU −1984，其余为0。对任意非负权重，慢镜像主成本不大于快镜像；即使主成本打平，memoryTransactions次级仍偏向慢镜像。这是特征偏序缺口，不能靠继续调权修复。另一个mixed对17维主成本相同，但完整字典序已有codeSize次级破tie，不能误报为第五个完整选择错误。

新增第18项`global_load_issue_groups`：从实际发射顺序统计连续f32全局请求组，浮点算术、Shuffle、Shared通信、同步或全局写打断分组；地址计算、字节组装和常量mov不打断。它不是eager/demand模式标签。回归在相同eager图中只移动独立V-load位置、保持图变量ID和前17项统计不变，组数从1变2。已知mixed两个镜像完整地址族和PTX opcode直方图相同，但首个浮点算术前为8/96条byte LDG，内核到EXIT静态SASS为911/696条；组数16/1能区分发射和地址复用机会。该指标仍是结构代理，不能当硬件并发度、动态指令数或单一因果证明。

## 同预算搜索与充分知情局部基线

新增可关闭的整图构造witness，再运行beam与refinement。构造、合法性探测和完整评分全部在总65536、refinement256之内；witness最多使用入口剩余预算的1/8，且不超过8192。最终模型成本不劣于自身实际构造并验证的witness；不保证全局最优，也不保证任意LI赋值均被包含。预算耗尽保留合法incumbent并如实标记非穷尽，cost failure和Unknown不伪装成无解。

LI拥有本producer的算术、访存和必要incoming repair成本，不使用未来消费者需求；其虚拟执行模式、顺序与scratch仅由自身backward cone决定。多个root按IR顺序规范化。LI从局部目标转回完整目标时首次完整评分也扣同一预算，诊断证书重评分单独说明。失败沿用原五键failure协议，并保存真实已耗预算和完整失败记录旁证。

## 独立审计与资源事实

C++账本通过可选SSA/约束lineage映射到导出程序。独立Python审计从实际128线程查询、PTX发射、所有者与地址重建18项特征，核对每SSA查询集合、每边通信和声明scratch。仅数目相同而查询内容不同也拒绝；篡改任一计价维度或同时篡改模型与账本的发射分组，都有拒绝证据。

修复包括receiver mux费用、常量mov、未访问但已声明的Shared scratch，以及最终store的use endpoint。r3在新形状CPU生成中发现终端Shared转换漏计32查询/32768字节/2屏障/15488地址ALU/384写bank excess；原legacy评分已计此边，漏洞在ordered demand分析与其查询预算证明。失败保留，没有GPU执行或性能标签。修复通用fetch后保持原六形状，不删除该例。

GPR使用编译器与实际加载函数的精确反馈；静态query/liveness量不冒充物理寄存器预测。Shared区分结构计划、编译器含reserved的原始计数和driver用户分配；LOCAL/STACK不推断spill数量。容量residency只是理论上限；本实验grid只有一CTA，不能把理论blocks/SM当实测occupancy。

## 创新范围与借鉴边界

借鉴已有编译器的增量存活工作计费、硬资源约束和预算内保留合法incumbent等原则。GF(2)、核/像/正交补、广播消除和beam本身不是Frisk独有发明，也没有复制Triton的相对周期常数。

本轮Frisk实现的组合是：跨SSA消费者仿射需求传播、含非零相关复制列的最小同线程截面、保留真实查询重数、有界通信物化、发射顺序费用、图级联合选择，以及逐项可反证的后端执行审计。全部模型/搜索收益必须由固定算法、ABI、精度与预算的实际测量支持；本报告不构成优于Triton/TileLang的外部对比。

支持域仍为单CTA、128线程、dyadic f32纯DAG、splat常量、单最终store和C转换策略，并受公开query/载体/Shared上限约束。MMA/Attention/控制流/O采用明确carrier近似回退，不能因本轮纯DAG通过而宣称这些分支也完成了需求模型校准。旧15/17项profile、格式和模型ID保留；新18项使用独立v2格式/v5d语义ID。


## 冻结版本和证据位置

- 源码recipe：`cb7dbb165812c20b4a1efd250923dc2965ff500ab54ff4dd7017b6fbf78faa24`，405文件；全源快照515文件及5个二进制，HEAD `9f79155ac08f82133f4ddb6554af552feefca8d2`，patch/archive重放通过。
- profile字节SHA：`1de84295d884f8a54fd2d78d85ae785d6a81f15e9c08790ae1007b9f51efbf23`。
- 模型ID：`sm90-layout-calibrated-v5d-adae66dfbc54b06154450d15309f43d4-leg`；原始JSON SHA与C++规范化语义SHA不同是有意设计，两链均已核对。
- registration SHA：`51634e2bbec71202af5926471e5ade4a1a8c588f2728649be357ebf732090dd1`。
- protocol SHA：`062c90d335a025fc93e8d20cc3e30f7dd5c038dda669bdeb4585f52d9a0f23cd`。
- measurement freeze/completion SHA：`10e7f680b5f05a288824b19ffba8ac0aa68086b2654a6574af0ca20e4495557b` / `6711e6958cfa78958275b10113608582f7259b201729e856f46d92afcb6aabef`。

主运行：`guoqiao/optimize_plan/demand_cost_search_repair_artifacts/run-demand-repair-r4-20261103/`。其中`quality-summary.json`、`group-quality.json`、`oracle-quality.json`、`resource-quality.json`、`time-proxy-quality.json`、`search-quality.json`和`measurement/`为完整原始/分析结果。

独立证书包括`development/root-validation-audit-r4.json`、`root-measurement-audit-r4.json`、`independent-statistical-review-r4.json`、`r4-frozen-v5d-union-crossscore.json`以及主运行`resource-review-r4.json`。`development/source-change-attribution.json`对比上一阶段：19新增/27修改/0删除源码；旧未提交阶段未重置。

所有失败/被替代CPU候选和launcher失败原样留存。r4启动器一次空解释器路径错误在worker启动前发生，恢复只改可变manifest的status与error；114生成行和全部生成文件哈希不变，旧failed manifest、日志和恢复证明完整。未把它当布局/求解失败。

默认仍为legacy-v2；seed默认关闭，新模型仅显式opt-in，本报告没有提交或推送代码，也未开展Triton/TileLang外部公平对比。最终证据索引保存在本阶段`evidence-index.json`，包含文档快照与所有失败历史。

最终posthoc诊断：主运行`failure-diagnosis-r4.json`（SHA256 `984a16fbcd14653ca2a015b92b675a3fd352000c9e0351c52311b3024ce190b2`）绑定四镜像和重新反汇编的SASS，`binding_errors=[]`。exp的load-group项抵消了部分坏偏好，而不是单独造成失败；conflict中该项确实翻转排序。未用任何posthoc发现改写本轮模型、结果或门槛。
