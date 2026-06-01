# 休闲零食连锁店商品销量预测

## 摘要

本文针对休闲零食连锁店商品销量预测问题，基于历史零售明细和天气、节假日、促销活动数据，构造了日期-门店-商品粒度的日销量面板。数据清洗阶段确认销售数据覆盖 2020-04-01 至 2022-03-31，包含 7 家门店、12 个商品代码和 8 个商品类别；负销量按人工确认为损耗或冲销类负向调整，不作为正常购买需求，预测目标采用正向销量 `positive_sales`。

针对四个问题，本文采用可解释、可复现的分层建模路线。问题一使用移动平均、同星期均值和简单指数平滑分析门店、商品和门店-商品组合销量，门店层面以 7 日移动平均为主，商品和门店-商品层面以简单指数平滑为主，并补充 7 日窗口总量验证。问题二以附件 `category` 字段作为同类零食的正式聚合依据，使用 Pearson、Spearman、去星期效应相关和门店内稳定性检查分析商品同步波动，类别预测采用“类别聚合后直接预测 + 简单指数平滑”。问题三以“合并天气 + `log1p` 销量 + 历史控制固定效应回归”为主分析，描述性统计、温度分箱、去异常日、分门店/分类别稳健性和随机森林置换重要性作为辅助。问题四将历史销量、类别和外部变量纳入 Ridge 综合模型，并以严格 7 日递推验证作为主评价口径。

结果显示，问题一日滚动验证下门店主模型 WAPE=37.49%，商品主模型 WAPE=38.14%；问题二类别聚合后直接预测 WAPE=34.74%；问题三主回归 WAPE=84.53%。问题四严格 7 日递推验证下，问题一门店-商品简单指数平滑 WAPE=76.35%，综合 Ridge WAPE=76.68%。进一步的消融与低销量鲁棒性检验表明，低销量序列使用指数平滑、常规序列使用综合 Ridge 的混合策略 WAPE=75.85%，较两个单一模型略有改善。有限优化中近 28 日均值兜底候选在同一验证集上 WAPE 更低，但留一窗口检查未达到稳定替换阈值，因此本文仍将低销量指数平滑-常规 Ridge 混合策略作为候选最终预测方案。预测表 `outputs/final_7day_forecast_hybrid_low_volume.csv` 是在历史同期天气和活动日情景下得到的条件预测，原 `outputs/final_7day_forecast.csv` 保留为完整 Ridge 对照。

**关键词**：销量预测；移动平均；指数平滑；固定效应回归；严格递推验证；WAPE

## 1 问题重述

### 1.1 背景

休闲零食连锁门店需要根据历史销售数据安排订货、库存和促销活动。不同门店的客流基础不同，不同商品的销量规模和波动性也不同；同时，天气、节假日和门店活动可能与销售变化相关。因此，本题需要在理解历史销量结构的基础上，给出未来 7 天的销量预测，并分析商品关联和外部因素。

### 1.2 需要解决的问题

问题一要求分析不同门店和不同零食的销量情况，并分别预测未来 7 天总销量。该问题的输入是历史销售明细，输出包括门店层面、商品层面和门店-商品组合层面的预测结果。

问题二要求分析同一门店内不同零食销量之间的联系，整合同类零食，并预测未来 7 天总销量。该问题的关键是区分“商品销量同步波动”和“真实业务类别聚合”：相关性用于分析联系，附件 `category` 字段用于正式类别预测。

问题三要求分析天气、节假日、活动日等因素与销量之间的关系。由于数据是观察数据而非随机实验，本文只讨论统计关联，不把外部变量写成严格因果影响。

问题四要求结合前三问模型和结论，预测各门店各种零食未来 7 天总销量，并与前序模型比较误差。该问题的关键是使用与未来 7 天预测一致的严格递推验证口径，避免把日滚动一步验证误写成未来 7 天一次性预测效果。

### 1.3 数据说明

附件一为历史零售明细，原始表含 67084 行、14 列，字段包括门店编号、门店名称、商品代码、商品名称、数量、销售额、收款时间和类别名称等。附件二为天气、节假日和促销活动数据，含 760 行、9 列，字段包括日期、天气、最高温、最低温、风力、节日、活动日和工休日等。数据字典见 `DATA_DICTIONARY.md`。

## 2 问题分析

本题的核心不是单一模型选择，而是按题目四问建立互相衔接的分析框架。问题一先建立可解释的销量 baseline，形成门店、商品和门店-商品三个粒度的基础预测；问题二在商品关联分析基础上进行类别聚合，解决同类零食整合预测；问题三分析外部变量与销量之间的条件统计关联，为综合模型提供特征解释；问题四在门店-商品粒度上融合历史销量、日历、类别和外部变量，并用严格 7 日递推验证检验其是否超过强 baseline。

预测评价不能随机切分，因为随机切分会让未来日期信息进入训练过程。本文使用时间顺序切分、日滚动验证和严格 7 日递推窗口验证。日滚动一步验证评价“每天更新真实销量后预测下一天”的能力；严格 7 日递推验证评价“窗口开始时一次性向未来预测 7 天”的能力，两者不能混用。

## 3 模型假设

假设一：题目提供的销售明细和外部变量记录总体真实可靠。对于已发现的数据问题，本文采用记录、标记和统一口径处理，而不是直接删除原始数据。

假设二：短期内历史销售规律具有一定延续性。未来 7 天预测依赖最近历史销量、同星期规律和类别/门店稳定差异。

假设三：负销量不代表顾客正常购买需求。根据人工确认，负销量多对应破损、过期或冲销类负向调整，因此预测目标采用 `positive_sales`。

假设四：附件类别字段可以作为同类零食的正式聚合依据。相关性聚类和销量规模分组只作为补充分析，不替代业务类别。

假设五：天气、节假日和活动日与销量之间可以存在统计关联，但当前数据不足以识别严格因果效应。本文所称“影响因素分析”均指统计关联分析。

假设六：未来 7 天无附件真实天气和真实活动安排。最终预测若使用天气和活动变量，应解释为“给定历史同期外部变量情景下”的条件预测。

## 4 符号说明

表 1：主要符号说明。

| 符号 | 含义 |
|---|---|
| $t$ | 日期序号 |
| $s$ | 门店编号 |
| $p$ | 商品编号 |
| $g$ | 商品类别 |
| $y_{s,p,t}$ | 门店 $s$、商品 $p$ 在日期 $t$ 的正向销量 |
| $\hat y_{s,p,t}$ | 对 $y_{s,p,t}$ 的预测值 |
| $Y_{t,g}$ | 日期 $t$ 类别 $g$ 的聚合销量 |
| $S_t$ | 指数平滑中的平滑水平 |
| $\alpha$ | 指数平滑系数 |
| $r_{pq}$ | 商品 $p$ 与商品 $q$ 的销量相关系数 |
| $X_t$ | 日历、天气、活动日等外部变量 |
| $MAE$ | 平均绝对误差 |
| $RMSE$ | 均方根误差 |
| $WAPE$ | 加权绝对百分比误差 |

## 5 数据预处理

### 5.1 数据检查

销售数据日期范围为 2020-04-01 至 2022-03-31，含 7 家门店、12 个商品代码、13 个商品名称和 8 个类别。商品代码 `11001020` 对应两个海苔商品名称，经人工确认以商品编号为准，名称差异视为录入笔误。天气数据日期范围为 2020-03-01 至 2022-03-30，不能覆盖销售最后一天 2022-03-31 和未来 2022-04-01 至 2022-04-07。

数据质量检查发现：销售明细中完全重复行 16605 条，经人工确认保留；负销量记录 54 条，作为损耗/冲销类负向调整；交易级数量范围为 -24 至 2712，IQR 规则标记 5536 条交易级极端记录。本文不删除原始记录，而是在处理表中保留正向销量、负向调整和异常标记。

### 5.2 日销量面板构造

本文将交易明细聚合为日期-门店-商品日销量面板，并补齐出现过的门店-商品组合在完整日期上的零销量记录。输出表 `data/processed/daily_store_product_sales.csv` 含 59130 行，日期范围为 2020-04-01 至 2022-03-31。合并外部变量后得到 `data/processed/modeling_base_table.csv`，同样为 59130 行。

日销量字段包括：

- `daily_sales`：日净销量，包含负向调整；
- `positive_sales`：正向销售数量，作为主预测目标；
- `negative_adjustment_qty`：负向调整数量绝对值；
- `has_external_data`：是否成功匹配附件二外部变量。

### 5.3 外部变量处理

附件二的 `温度` 和 `温度.1` 经人工确认为最高温和最低温，分别记为 `max_temperature` 和 `min_temperature`；`活动日` 确认为门店促销活动日；`工休日` 表示周末/节假日等休息日。缺少外部变量的销售日期为 2022-03-31，对应合并后 81 行 `has_external_data=0`。未来 7 天没有附件真实天气和活动日观测，本文在问题四中采用历史同期情景，并明确其条件预测性质。

## 6 模型建立与求解

### 6.1 问题一：门店与商品销量分析及未来 7 天预测

问题一以正向销量为目标，分别在门店、商品和门店-商品层级建模。设某一序列第 $t$ 天销量为 $y_t$。

7 日移动平均模型为：

$$
\hat y_{t+1}=\frac{1}{7}\sum_{i=0}^{6}y_{t-i}.
$$

它用最近 7 天平均销量代表短期水平，适合作为门店层面的主模型。

同星期均值模型为：

$$
\hat y_t=\frac{1}{m}\sum_{j=1}^{m}y_{t-7j},\quad m\le 8.
$$

它用历史相同星期几销量预测目标日，作为检验星期效应的 baseline。

简单指数平滑模型为：

$$
S_t=\alpha y_t+(1-\alpha)S_{t-1},\quad \hat y_{t+1}=S_t,\quad 0<\alpha\le 1.
$$

该模型更重视近期销量，适合作为商品和门店-商品层面的主模型。

阶段 2 日滚动验证结果见表 2：

表 2：问题一日滚动验证下主模型误差。

| 层级 | 主模型 | MAE | RMSE | WAPE |
|---|---|---:|---:|---:|
| 门店 | 移动平均(7日) | 9.685 | 15.445 | 37.49% |
| 商品 | 简单指数平滑 | 5.748 | 10.968 | 38.14% |
| 门店-商品 | 简单指数平滑 | 1.783 | 3.733 | 74.95% |

为了使评价更贴近“未来 7 天总销量”，方法复核中补充 7 日窗口总量验证。结果显示，原问题一主方案在商品层面 WAPE=21.51%，门店层面 WAPE=22.12%，门店-商品层面 WAPE=43.06%。更复杂的滞后 Ridge 和随机森林未形成稳定优势，因此问题一保留原主模型。该判断依据见 `outputs/method_search/final_method_selection.md`。

### 6.2 问题二：商品关联分析与类别聚合预测

对每个门店构造日期 $\times$ 商品销量矩阵：

$$
Y_s=(y_{t,p}),\quad t=1,\ldots,n,\ p=1,\ldots,m.
$$

商品 $p$ 与商品 $q$ 的 Pearson 相关系数为：

$$
r_{pq}=\frac{\sum_t(y_{t,p}-\bar y_p)(y_{t,q}-\bar y_q)}
{\sqrt{\sum_t(y_{t,p}-\bar y_p)^2}\sqrt{\sum_t(y_{t,q}-\bar y_q)^2}}.
$$

为避免单一 Pearson 相关造成过度解释，本文补充 Spearman 相关、去星期效应 Pearson 和门店内相关稳定性检查。商品关联结论只解释为同步波动，不写成因果或替代关系。

类别聚合按附件 `category` 字段进行：

$$
Y_{t,g}=\sum_{p:c(p)=g}y_{t,p}.
$$

比较“单品预测后按类别加总”和“类别聚合后直接预测”。验证结果见表 3：

表 3：问题二类别预测策略误差比较。

| 方法 | 模型 | MAE | RMSE | WAPE |
|---|---|---:|---:|---:|
| 类别聚合后直接预测 | 简单指数平滑 | 7.853 | 13.933 | 34.74% |
| 单品预测后按类别加总 | 简单指数平滑 | 7.873 | 13.949 | 34.83% |
| 类别聚合后直接预测 | 移动平均(7日) | 8.246 | 14.174 | 36.48% |
| 类别聚合后直接预测 | 同星期均值(近8周) | 8.851 | 15.057 | 39.15% |

因此，问题二主方案采用“附件类别聚合后直接预测 + 简单指数平滑”。相关性聚类 WAPE=31.02%，销量规模分组 WAPE=24.24%，但前者业务类别含义弱，后者预测对象变成高/中/低销量组，均不替代附件类别主方案。

### 6.3 问题三：外部因素统计关联分析

问题三主模型采用“合并天气 + `log1p` 销量 + 历史控制固定效应回归”。设 $z_{s,p,t}=\log(1+y_{s,p,t})$，模型形式为：

$$
z_{s,p,t}=\beta_0+\beta^\top External_t+\rho_1 lag7_{s,p,t}
+\rho_2 rolling28_{s,p,t}+\gamma_s+\delta_p+\eta_w+\mu_m+\varepsilon_{s,p,t}.
$$

其中，$External_t$ 包括合并天气类别、平均温度、昼夜温差、风力、节假日和活动日；$\gamma_s$、$\delta_p$、$\eta_w$、$\mu_m$ 分别表示门店、商品、星期和月份固定效应；`lag_7` 和 `rolling_28_prev` 只使用过去销量。

选择该模型的原因是：详细天气类别中存在雨夹雪、暴雨等少数样本天气，直接排序容易不稳；原始销量分布偏斜，对极端销量敏感；加入历史控制项后能减少基础需求差异对外部变量系数的干扰。

方法探索验证指标见表 4：

表 4：问题三外部因素分析方法验证指标。

| 方法 | MAE | RMSE | WAPE | 论文定位 |
|---|---:|---:|---:|---|
| 原方案详细天气 FE 回归 | 1.969 | 4.144 | 88.09% | 补充 |
| 合并天气 + 对数销量 + 历史控制 FE 回归 | 1.890 | 4.448 | 84.53% | 主分析 |
| 温度分箱 + 对数销量 + 历史控制 FE 回归 | 1.889 | 4.445 | 84.51% | 非线性检查 |
| 去异常日 + 对数销量 + 历史控制 FE 回归 | 1.825 | 4.254 | 84.94% | 稳健性检查 |
| 随机森林 grouped weather + history | 1.892 | 4.178 | 84.66% | 预测贡献补充 |

主模型系数显示，活动日变量与销量上升呈较稳定统计关联；在分门店稳健性中，活动日系数为正且显著的门店数为 5/7，在分类别稳健性中为 7/8。天气条件也存在一定统计关联，但具体天气类别差异受样本量限制。本文不分析湿度，因为附件中没有湿度字段。

### 6.4 问题四：综合预测模型与严格递推验证

问题四在门店-商品粒度上融合前三问信息。综合 Ridge 的特征包括滞后销量、滚动均值、星期、月份、是否周末、门店、商品、类别、天气、节假日和活动日。设 $x_{s,p,t}$ 为上述变量经数值化和独热编码后的特征向量，模型可写为：

$$
\hat y_{s,p,t}=x_{s,p,t}^{\top}\hat\beta,\quad
\hat\beta=\arg\min_{\beta}\left\{\sum_{(s,p,t)\in \mathcal T}(y_{s,p,t}-x_{s,p,t}^{\top}\beta)^2+\lambda\sum_{j=1}^{d}\beta_j^2\right\}.
$$

其中，$\mathcal T$ 为训练样本集合，$\lambda$ 为正则化强度，$d$ 为特征维数。$Lag$ 包括 `lag_1`、`lag_7`、`lag_14`，$Roll$ 包括 `rolling_mean_7` 和 `rolling_mean_14`，$Cal$ 包括星期、月份和周末标记，$External$ 包括天气、温度、风力、节假日和活动日。Ridge 回归用于可解释的综合线性预测；随机森林和其他树模型只作为补充比较，不作为论文主模型。

信息泄露专项检查 `outputs/q4_information_leakage_review.md` 指出：原日滚动验证没有当天销量直接进入当天特征，但验证窗口后续日期的 `lag_1` 和滚动均值会使用窗口内前几天真实销量。因此，问题四以 `outputs/q4_recursive_7day_model_comparison.md` 中的严格 7 日递推验证作为主口径。

严格递推验证窗口见表 5：

表 5：问题四严格 7 日递推验证窗口。

| 窗口起点 | 窗口终点 | 长度 |
|---|---|---:|
| 2022-03-01 | 2022-03-07 | 7 |
| 2022-03-08 | 2022-03-14 | 7 |
| 2022-03-15 | 2022-03-21 | 7 |
| 2022-03-22 | 2022-03-28 | 7 |

严格递推下，门店-商品层级误差见表 6：

表 6：问题四严格 7 日递推主粒度误差比较。

| 模型 | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| 问题一门店-商品简单指数平滑 | 1.816 | 3.772 | 76.35% |
| 综合 Ridge 回归 | 1.824 | 3.702 | 76.68% |
| 14 日滚动均值 baseline | 1.847 | 3.819 | 77.66% |
| 7 日移动平均 baseline | 1.865 | 3.797 | 78.43% |
| 综合随机森林 | 1.988 | 4.842 | 83.60% |

因此，综合 Ridge 在严格递推下优于问题四内部 baseline，但未超过问题一门店-商品简单指数平滑强 baseline。本文将综合 Ridge 定位为融合历史销量、类别和外部变量的综合解释模型，而不写成“严格 7 日预测显著改进”。

在此基础上，本文补充严格递推消融实验。完整综合 Ridge 的 WAPE 为 76.68%，去天气 Ridge 为 77.73%，去活动日 Ridge 为 77.22%，无销量历史 Ridge 为 84.49%。该结果说明销量滞后和滚动统计仍是核心信息，天气和活动日具有一定预测补充作用，但不能解释为因果影响。消融图见 `figures/q4_ablation_store_product_wape.png`，完整报告见 `outputs/q4_ablation_report.md`。

低销量鲁棒性检验进一步显示，低销量序列使用指数平滑、常规序列使用综合 Ridge 的混合策略 WAPE 为 75.85%，优于问题一简单指数平滑的 76.35% 和完整综合 Ridge 的 76.68%。低销量判定只使用预测窗口开始日前历史数据：正销量天数占比不高于 0.15 或历史日均销量低于 0.20 时记为 `very_sparse`；零销量比例不低于 0.75 或历史日均销量低于 0.50 时记为 `low_volume_intermittent`；无正销量历史序列单独记为 `no_positive_history`。该改进来自对稀疏门店-商品组合的分层处理，而不是复杂模型堆叠。完整报告见 `outputs/q4_low_volume_strategy_report.md`。

本轮进一步尝试了不同低销量阈值、近 28 日均值、商品均值、类别均值和层级收缩兜底，并检查门店、商品和类别层级比例校准。全样本严格递推下，“低销量用近 28 日均值”的探索性候选 WAPE 可降至 74.95%，但留一窗口选模后的平均改善仅 0.19 个百分点，未达到 0.3 个百分点的稳定替换阈值。因此，本文不把该探索候选替换为主方案，只将其作为附加稳健性检查。详细结果见 `outputs/q4_candidate_model_review.md`。

天气敏感性检验 `outputs/q4_weather_sensitivity_report.md` 显示，完整外部变量 Ridge 的严格递推 WAPE 为 76.68%，去除天气变量 Ridge 为 77.73%，历史同期天气情景 Ridge 为 76.57%。因此，天气变量可作为补充情景特征，但不能写成稳定因果因素。误差诊断 `outputs/q4_error_diagnosis_report.md` 显示，综合 Ridge 在门店 `A地双桥路店`、低销量商品和样本不足的活动日场景误差较大；间歇性需求审查 `outputs/intermittent_demand_review.md` 显示，76 条门店-商品序列中有 54 条零销量比例不低于 50%，解释了门店-商品层级 WAPE 较高的原因。

## 7 结果分析

### 7.1 问题一结果

历史累计销量最高的门店为 `B地江山店店`，累计销量 21773；历史累计销量最高的商品为 `红牛`，累计销量 31119。门店和商品销量规模差异明显，说明不能只用统一平均水平预测所有对象。

未来 7 天门店预测中，`B地江山店店` 预测总销量最高，为 309.0；`C地五乡镇店` 为 280.0；`A地天九街店` 为 245.0。商品预测中，`乡吧哥蜜汁鸡翅` 预测 389.613，`红牛` 预测 286.719，`百事可乐600ml` 预测 136.219。问题一门店层面和商品层面分别建模，预测总量没有强制一致：门店层面合计 1296.000，商品层面合计 1226.006。

图 1 `figures/q1_store_total_sales.png` 和图 2 `figures/q1_product_total_sales.png` 用于证明门店和商品销量规模差异；图 3 `figures/q1_daily_total_trend.png` 用于展示总体日销量波动；图 4 `figures/q1_weekday_effect.png` 用于观察星期效应；图 5 `figures/q1_model_wape_comparison.png` 用于比较模型误差。

### 7.2 问题二结果

商品关联分析表明，全部门店汇总层面存在 2 对强正相关商品对和 25 对弱相关商品对。强正相关只能说明历史日销量同步波动，可能共同受客流、消费场景或促销节奏影响，不能证明商品之间存在直接带动关系。负相关候选的相关系数绝对值较小，当前数据未发现强替代关系证据。

类别预测结果显示，未来 7 天预测最高的类别为 `包装散称`，预测 389.613；其次为 `功能饮料`，预测 286.719；`碳酸饮料` 预测 247.326。类别聚合可以降低单品零销量和偶然大单带来的波动，但会损失同一类别内部商品差异。

图 6 `figures/q2_overall_product_correlation_heatmap.png` 用于展示商品同步波动结构；图 7 `figures/q2_category_daily_trend.png` 用于展示类别销量趋势；图 8 `figures/q2_category_method_wape_comparison.png` 用于比较类别聚合后直接预测和单品加总策略。

### 7.3 问题三结果

描述性统计显示，不同天气、节假日、周末和活动日下的平均销量存在差异，但这些差异可能与门店、商品、星期和月份同时相关。固定效应回归和稳健性检查表明，活动日变量与销量上升之间的统计关联相对更稳定；天气变量存在一定关联，但细分天气类别受样本量限制；节假日和周末与星期效应、月份季节性重叠，结论应作为辅助。

随机森林置换重要性显示，历史销量水平、活动日、商品和类别等变量对预测误差有贡献，但置换重要性不能说明方向，更不能证明因果。图 9 `figures/q3_binary_factor_mean_sales.png` 用于展示节假日、周末、活动日的描述性均值差异；图 10 `figures/q3_regression_factor_effect_ranking.png` 用于展示回归可比关联强度；图 11 `figures/q3_random_forest_permutation_importance.png` 用于展示非线性模型中的预测贡献。

### 7.4 问题四结果

严格 7 日递推验证显示，综合 Ridge 的 WAPE 为 76.68%，略高于问题一门店-商品简单指数平滑的 76.35%，但低于 14 日滚动均值 baseline 的 77.66% 和 7 日移动平均 baseline 的 78.43%。因此，综合 Ridge 没有在主粒度上超过最强简单模型，但其优势是把历史销量、类别、天气、节假日和活动日纳入统一解释框架。

消融与低销量鲁棒性检验表明，低销量组合更适合采用问题一简单指数平滑兜底，常规组合保留综合 Ridge。该混合策略在严格递推主粒度 WAPE=75.85%，相对问题一简单指数平滑下降 0.49 个百分点，相对完整综合 Ridge 下降 0.83 个百分点。由于验证窗口只有 4 个完整 7 日窗口，本文将该结果写作稳健性改进，而不写成显著突破。

未来 7 天低销量混合策略条件预测中，各门店预测总销量最高的是 `B地江山店店` 320.342，其次为 `C地五乡镇店` 301.763 和 `A地天九街店` 227.846；各类别预测最高的是 `包装散称` 439.965，其次为 `功能饮料` 364.481 和 `碳酸饮料` 240.589。完整门店-商品-日期小数预测表见 `outputs/final_7day_forecast_hybrid_low_volume.csv`；按“件数”提交时使用整数化副本 `outputs/final_7day_forecast_hybrid_low_volume_integer.csv`。原完整 Ridge 小数预测表 `outputs/final_7day_forecast.csv` 保留为对照。

图 12 `figures/q4_hybrid_forecast_7day_total_by_store.png` 用于展示低销量混合策略下未来 7 天门店预测总量；图 13 `figures/q4_hybrid_forecast_7day_total_by_category.png` 用于展示同一预测表的类别预测总量；图 14 `figures/q4_error_by_store_bar.png` 和图 15 `figures/q4_error_by_category_bar.png` 用于定位误差较高的门店和类别；图 16 `figures/q4_actual_vs_predicted_scatter.png` 用于检查严格递推验证中预测值与真实值的对应关系；图 17 `figures/q4_ablation_store_product_wape.png` 用于展示消融结果；图 18 `figures/q4_low_volume_strategy_wape.png` 用于展示低销量混合策略比较；图 19 `figures/final_forecast_interval_by_store.png` 用于展示基于严格递推残差的门店级经验预测区间。

误差诊断表明，商品层面按验证期销量中位数划分时，高销量商品组 WAPE=19.23%，低销量商品组 WAPE=33.07%；但在门店-商品低销量序列口径下，当前混合策略的低销量序列整体 WAPE 约为 122.53%。这两个数值对应不同聚合口径，不能混用。门店 `A地双桥路店` WAPE=116.93%，说明低销量门店或稀疏序列的百分比误差会显著放大。该结论与 `outputs/intermittent_demand_review.md` 中“54 条门店-商品序列零销量比例不低于 50%”一致。

## 8 模型评价

### 8.1 优缺点分析

本文优点在于：第一，数据口径清晰，区分正向销量、净销量和负向调整；第二，模型与题目四问一一对应，形成门店、商品、类别和门店-商品的分层预测框架；第三，主模型优先采用移动平均、同星期均值、指数平滑和固定效应回归，公式简单、可解释、易复现；第四，对综合模型进行了信息泄露审查，并以严格 7 日递推验证作为主口径；第五，对外部变量和商品关联均设置了证据边界，没有把相关性写成因果性。

模型局限包括：第一，门店-商品粒度存在大量零销量和间歇性需求，导致 WAPE 较高；第二，未来真实天气和活动安排不可得，含外部变量预测依赖历史同期情景；第三，附件没有顾客购物篮、价格实验、库存和陈列信息，因此无法识别商品互补、替代或促销因果效应；第四，验证期只有有限的完整 7 日窗口，显著性检验只能作为辅助证据。

### 8.2 不确定性与改进方向

不确定性主要来自四方面：外部情景假设、低销量序列误差放大、稀有天气样本不足、以及模型选择与验证样本重叠。本文用当前混合策略在 4 个严格 7 日递推验证窗口上的 7 日聚合残差构造经验 80% 区间，输出 `tables/final_forecast_interval_by_store.csv`、`tables/final_forecast_interval_by_product.csv` 和 `tables/final_forecast_interval_by_category.csv`；该区间只是历史递推误差范围，不是严格置信区间。若正式提交要求整数销量，本文另生成 `outputs/final_7day_forecast_hybrid_low_volume_integer.csv`：预测值小于 0 的先置为 0，极小预测值低于 `1e-6` 时置为 0，其余按四舍五入转为整数；原始小数预测表不覆盖，继续作为模型输出依据。原完整 Ridge 整数化文件 `outputs/final_7day_forecast_integer.csv` 同步保留用于复核。

## 9 结论

本文完成了休闲零食连锁店商品销量预测的四问建模。问题一保留移动平均、同星期均值和简单指数平滑，建立了门店、商品和门店-商品的基础预测；问题二以附件类别字段整合同类零食，商品相关性分析作为同步波动证据，类别预测主方案为类别聚合后直接预测；问题三采用合并天气、对数销量和历史控制固定效应回归，得到外部变量与销量之间的条件统计关联；问题四采用综合 Ridge 模型融合前三问信息，并用严格 7 日递推验证检验未来 7 天预测能力。

最终结论是：简单可解释模型在本题中具有很强竞争力。问题四综合 Ridge 虽然在日滚动一步预测和部分聚合层级上有数值优势，但在严格 7 日递推主粒度下未超过问题一门店-商品简单指数平滑强 baseline。因此，本文不夸大单一综合模型的预测提升，而将其作为融合历史销量、类别和外部变量的综合解释框架。进一步的低销量鲁棒性检验表明，对低销量组合采用指数平滑、对常规组合采用综合 Ridge 的混合策略在严格递推主粒度上 WAPE=75.85%，略优于两个单一模型。有限优化中近 28 日均值兜底候选在同一验证集上表现更低，但留一窗口检查未达到稳定替换阈值，因此本文仍推荐使用 `outputs/final_7day_forecast_hybrid_low_volume.csv` 作为候选最终预测表，同时保留 `outputs/final_7day_forecast.csv` 作为完整 Ridge 对照；两者均是在历史同期天气和活动日情景下的条件预测。混合策略整数化提交副本为 `outputs/final_7day_forecast_hybrid_low_volume_integer.csv`。

## 参考文献

[1] Hyndman R J, Athanasopoulos G. Forecasting: Principles and Practice. 3rd ed. OTexts, 2021.

[2] Montgomery D C, Peck E A, Vining G G. Introduction to Linear Regression Analysis. 5th ed. Wiley, 2012.

[3] Hoerl A E, Kennard R W. Ridge Regression: Biased Estimation for Nonorthogonal Problems. Technometrics, 1970, 12(1): 55-67.

[4] Breiman L. Random Forests. Machine Learning, 2001, 45: 5-32.

[5] Croston J D. Forecasting and Stock Control for Intermittent Demands. Operational Research Quarterly, 1972, 23(3): 289-303.

## 附录

### 附录 A 主要代码文件

- 数据读取与清洗：`src/data_loader.py`、`src/preprocessing.py`
- 特征工程：`src/features.py`
- 问题二分析：`src/stage3_q2_analysis.py`
- 问题三分析：`src/stage4_q3_analysis.py`
- 问题四综合模型：`src/stage5_q4_analysis.py`
- 严格递推验证：`src/stage5_q4_recursive_validation.py`
- 问题四误差诊断：`src/stage5_q4_error_diagnosis.py`
- 天气敏感性分析：`src/stage5_q4_weather_sensitivity.py`
- 严格递推消融检验：`src/stage5_q4_ablation.py`
- 低销量鲁棒性策略：`src/stage5_q4_low_volume_strategy.py`
- 低销量混合最终预测：`src/stage5_q4_hybrid_final_forecast.py`
- 间歇性需求审查：`src/intermittent_demand_review.py`

### 附录 B 主要结果文件

- 数据检查：`outputs/stage1_data_inspection_report.md`
- 问题一报告：`outputs/stage2_q1_modeling_report.md`
- 问题二报告：`outputs/stage3_q2_relationship_report.md`
- 问题三报告：`outputs/stage4_q3_factor_analysis_report.md`
- 问题四报告：`outputs/stage5_q4_final_model_report.md`
- 信息泄露审查：`outputs/q4_information_leakage_review.md`
- 严格递推验证：`outputs/q4_recursive_7day_model_comparison.md`
- 误差诊断：`outputs/q4_error_diagnosis_report.md`
- 天气敏感性检验：`outputs/q4_weather_sensitivity_report.md`
- 严格递推消融检验：`outputs/q4_ablation_report.md`
- 低销量鲁棒性策略：`outputs/q4_low_volume_strategy_report.md`
- 混合策略最终预测说明：`outputs/q4_hybrid_final_forecast_report.md`
- 间歇性需求审查：`outputs/intermittent_demand_review.md`
- 最终方法选择：`outputs/method_search/final_method_selection.md`
- 小数预测表：`outputs/final_7day_forecast.csv`
- 整数预测表：`outputs/final_7day_forecast_integer.csv`
- 混合策略小数预测表：`outputs/final_7day_forecast_hybrid_low_volume.csv`
- 混合策略整数预测表：`outputs/final_7day_forecast_hybrid_low_volume_integer.csv`

### 附录 C 主要图表文件

- 图 1：`figures/q1_store_total_sales.png`，门店累计销量。
- 图 2：`figures/q1_product_total_sales.png`，商品累计销量。
- 图 3：`figures/q1_daily_total_trend.png`，总体日销量趋势。
- 图 4：`figures/q1_weekday_effect.png`，星期效应。
- 图 5：`figures/q1_model_wape_comparison.png`，问题一模型 WAPE 对比。
- 图 6：`figures/q2_overall_product_correlation_heatmap.png`，商品相关性热力图。
- 图 7：`figures/q2_category_daily_trend.png`，类别日销量趋势。
- 图 8：`figures/q2_category_method_wape_comparison.png`，类别预测方法 WAPE 对比。
- 图 9：`figures/q3_binary_factor_mean_sales.png`，二元外部因素均值差异。
- 图 10：`figures/q3_regression_factor_effect_ranking.png`，回归关联强度排序。
- 图 11：`figures/q3_random_forest_permutation_importance.png`，随机森林置换重要性。
- 图 12：`figures/q4_hybrid_forecast_7day_total_by_store.png`，低销量混合策略未来 7 天门店预测总量。
- 图 13：`figures/q4_hybrid_forecast_7day_total_by_category.png`，低销量混合策略未来 7 天类别预测总量。
- 图 14：`figures/q4_error_by_store_bar.png`，门店误差诊断。
- 图 15：`figures/q4_error_by_category_bar.png`，类别误差诊断。
- 图 16：`figures/q4_actual_vs_predicted_scatter.png`，严格递推验证真实值与预测值散点。
- 图 17：`figures/q4_ablation_store_product_wape.png`，严格递推消融检验。
- 图 18：`figures/q4_low_volume_strategy_wape.png`，低销量鲁棒性策略比较。

### 附录 D 核心代码

正式排版稿末尾应附核心 Python 代码。建议至少包含：

- `src/stage5_q4_ablation.py`
- `src/stage5_q4_low_volume_strategy.py`
- `src/stage5_q4_hybrid_final_forecast.py`

完整项目代码随仓库提交，避免在正文中重复粘贴过长脚本影响阅读。
