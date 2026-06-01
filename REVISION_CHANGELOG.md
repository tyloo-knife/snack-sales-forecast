# REVISION_CHANGELOG

本文档记录本轮系统修订中每个问题对应的改动、涉及文件或脚本，以及新增或变更的真实数字。所有数值均来自项目脚本重跑或已生成结果文件；未修改 `data/raw/` 和参赛人员信息页。

| Phase / 任务 | 原问题 | 改了什么 | 对应文件 / 脚本 | 新增或变更的数字 |
|---|---|---|---|---|
| Phase 1 A3 | 最终预测在门店、类别、商品等层级独立取整，合计不一致 | 改为只在门店--商品--日期最细粒度上整数化，再由该表汇总所有上层表 | `src/stage5_q4_hybrid_final_forecast.py`；`outputs/final_7day_forecast_hybrid_low_volume_integer.csv`；`outputs/q4_hybrid_forecast_7day_total_by_*.csv`；`tables/q4_hybrid_forecast_7day_total_by_*.csv`；`paper/latex/sections/q4_model.tex`；`paper/latex/sections/appendix.tex` | 连续预测总量 `1329.894`；整数化后总量 `1328`；门店、商品、类别、门店--商品汇总总量均为 `1328` |
| Phase 1 A4 | 问题三缺少真实回归系数表，WAPE 容易被误读为主评价 | 增加固定效应回归系数表，明确问题三以系数、显著性和调整 `R^2` 为主 | `src/stage4_q3_analysis.py`；`outputs/q3_regression_coefficients_external.csv`；`tables/q3_regression_coefficients_external.csv`；`paper/latex/sections/q3_model.tex` | 主回归样本量 `58517`；调整 `R^2=0.370`；活动日系数 `0.325`，聚类标准误 `0.039`，`p<0.001` |
| Phase 1 A5 | 问题四没有直接回答“误差是否显著改进” | 增加同验证集配对显著性检验和 bootstrap 置信区间 | `src/stage5_q4_recursive_validation.py`；`outputs/q4_significance_tests.csv`；`tables/q4_significance_tests.csv`；`paper/latex/sections/q4_model.tex` | Ridge vs 问题一基准：`76.35%` vs `76.68%`，`p=0.461`，CI `[-1.56, 1.51]`；混合策略 vs 基准：`75.85%` vs `76.35%`，`p=0.0503`，CI `[-0.72, 1.34]`，均未达统计显著改进 |
| Phase 2 | PNG 图内烧录标题与 LaTeX 图题编号冲突 | 移除最终插图的图内标题，改由 LaTeX `\caption` 统一管理 | `src/stage2_q1_figures.py`；`src/stage3_q2_analysis.py`；`src/stage4_q3_analysis.py`；`src/stage5_q4_*.py`；`figures/*.png`；`paper/latex/sections/*.tex` | 最终论文保留 `16` 个插图；不再引用被删的 WAPE 对比图 |
| Phase 3 B1 | 不同章节 WAPE 口径混杂 | 增加验证口径对照表，并在正文中区分日滚动一步、7 日窗口总量、严格 7 日递推 | `paper/latex/sections/problem_analysis.tex`；`paper/latex/sections/model_evaluation.tex` | 问题一 7 日窗口总量 WAPE：门店约 `22%`、商品约 `21%`、门店--商品 `43.06%`；问题四严格递推基准 `76.35%` |
| Phase 3 B7 | 缺少集中稳健性和灵敏度说明 | 增加稳健性汇总和指数平滑 `alpha` 灵敏度表 | `src/stage6_phase3_structure_analysis.py`；`outputs/q1_exp_smoothing_alpha_sensitivity.csv`；`tables/q1_exp_smoothing_alpha_sensitivity.csv`；`paper/latex/sections/model_evaluation.tex` | `alpha=0.1/0.3/0.5` 下 WAPE：门店 `36.70%/39.08%/42.10%`；商品 `38.14%/39.89%/42.21%`；门店--商品 `73.56%/76.50%/80.44%` |
| Phase 3 C1 | 四问关系不够直观 | 新增技术路线图 | `src/stage6_phase3_structure_analysis.py`；`figures/technical_route_phase3.png`；`paper/latex/sections/problem_analysis.tex` | 新增 `1` 张技术路线图 |
| Phase 4 B2/B3 | 摘要和问题四叙事偏模板化，综合模型未跑赢基准的解释不足 | 重写摘要、问题四模型定位、局限和总结，突出间歇需求结构性误差与条件预测 | `paper/latex/sections/abstract.tex`；`paper/latex/sections/q4_model.tex`；`paper/latex/sections/strengths_weaknesses.tex` | 问题四严格递推：问题一 SES `76.35%`，综合 Ridge `76.68%`，低销量混合 `75.85%` |
| Phase 4 B4 | 问题一描述性统计不足 | 增加门店和商品层累计销量、日均销量、CV、零销量占比、趋势斜率 | `src/stage7_phase4_text_evidence.py`；`outputs/q1_store_descriptive_phase4.csv`；`outputs/q1_product_descriptive_phase4.csv`；`paper/latex/sections/q1_model.tex` | A地双桥路店日均销量 `1.08`、零销量占比 `74.0%`；洽洽早餐每日坚果累计销量 `57`、零销量占比 `97.7%` |
| Phase 4 B5 | 问题二商品联系解释不够具体 | 增加强正相关和弱相关商品对表，并明确只解释为同步波动 | `src/stage7_phase4_text_evidence.py`；`outputs/q2_strong_positive_pairs_full_phase4.csv`；`outputs/q2_weak_correlation_pairs_full_phase4.csv`；`paper/latex/sections/q2_model.tex` | 强正相关商品对 `2` 对；弱相关商品对 `25` 对；强正相关阈值 `r>=0.50`，弱相关阈值 `|r|<=0.10` |
| Phase 4 B6 | 节假日、周末、工休日重叠没有量化 | 增加日历变量重叠表和活动日样本量警示 | `src/stage7_phase4_text_evidence.py`；`outputs/q3_calendar_overlap_phase4.csv`；`outputs/q3_activity_day_scope_phase4.csv`；`paper/latex/sections/q3_model.tex` | 建模销售期活动日 `24` 天；附件二全期活动日 `25` 天；节假日且工休日占节假日 `85.0%`；周末且工休日占周末 `95.7%` |
| Phase 5 C2 | WAPE 图表重复，问题四随机森林展开过多 | 删除正文中冗余 WAPE 图引用，保留表格；随机森林压缩为表中一行和一句解释；重写 9.2 为跨口径评价 | `paper/latex/sections/q1_model.tex`；`paper/latex/sections/q2_model.tex`；`paper/latex/sections/q4_model.tex`；`paper/latex/sections/model_evaluation.tex` | 删除正文引用的 `4` 张冗余 WAPE 图；保留随机森林 WAPE `83.60%` |
| Phase 5 A6 | 真实销量为 0 时 WAPE 无定义的说明不足 | 增加 WAPE 分母为 0 的处理说明，区分误差评价表和未来预测汇总表 | `paper/latex/sections/model_evaluation.tex`；`paper/latex/sections/appendix.tex` | 包装冲饮未来预测量按 `0` 列示，不计算 WAPE |
| Phase 6 C3 | 参考文献缺少间歇需求和预测评价指标补充 | 新增 2 篇正规文献，并在正文中回指 | `paper/latex/references.bib`；`paper/latex/sections/model_evaluation.tex`；`paper/latex/sections/q4_model.tex`；`paper/latex/sections/strengths_weaknesses.tex` | 新增 `Syntetos and Boylan (2005)`、`Hyndman and Koehler (2006)` 两条文献 |
| Phase 7 | 需要完整终检与交付记录 | 重跑完整数据/建模脚本，重新编译 LaTeX，补齐附录误差表验证口径，做一致性和占位符检查 | `RESULT_LOG.md`；`REVISION_CHANGELOG.md`；`paper/latex/sections/appendix.tex`；`paper/latex/sections/q2_model.tex`；`paper/latex/main.pdf` | 预处理后 `daily_store_product_sales` 与 `modeling_base_table` 均为 `59130` 行；最终整数预测共 `532` 行、总量 `1328`；`main.pdf` 编译成功 `44` 页；LaTeX 无未定义引用或引用缺失 |

## Phase 7 终检摘要

- 完整管线已重跑：预处理、问题一图表、问题二、问题三、问题四递推验证、误差诊断、天气敏感性、消融、低销量混合和最终预测均成功执行。
- 最终预测总量一致：门店、商品、类别、门店--商品、日期汇总均来自同一门店--商品--日期整数预测表，总量均为 `1328`。
- 图表检查通过：正文最终引用 `16` 张图，图文件均存在；图标签无重复、无缺失引用；最终阶段脚本未检出 `set_title`、`suptitle` 或 `plt.title`。
- WAPE 口径已显式标注：日滚动一步、7 日窗口总量、严格 7 日递推、分母为 0 的情形均在正文或表题中说明。
- 问题三和问题四核心证据已落地：问题三有真实固定效应回归系数表；问题四有配对显著性检验，结论保持“未达统计显著改进”。
- 待填标记检查通过：未检出常见待填标记、空引用或空文献引用。
- LaTeX 编译成功生成 `paper/latex/main.pdf`，共 `44` 页；命令输出仍有既有 `Object @page.1 already defined` 警告，不影响 PDF 生成。
