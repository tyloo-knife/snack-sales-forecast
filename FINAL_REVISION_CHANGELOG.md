# FINAL_REVISION_CHANGELOG

## Phase 1 代码硬伤

| 问题 | 改动 | 文件/脚本 | 是否影响数字 |
|---|---|---|---|
| `json.dumps` 末尾调用缺少导入 | 补充 `import json` | `src/stage3_q2_analysis.py`; `src/stage4_q3_analysis.py`; `src/stage5_q4_analysis.py` | 否；仅使 summary JSON 正常写出 |
| `q4_significance_tests.csv` 被日滚动和严格递推两种口径覆盖 | 日滚动旁路口径改写为 `q4_significance_tests_daily.csv`；严格递推口径继续使用 `q4_significance_tests.csv` | `src/stage5_q4_analysis.py`; `tables/q4_significance_tests_daily.csv`; `outputs/q4_significance_tests_daily.csv`; `paper/latex/sections/appendix.tex` | 否；仅拆分文件名和 schema |
| `stage6`/`stage7` 直接运行无法导入 `src.*` | 在导入 `src.*` 前加入项目根目录到 `sys.path` | `src/stage6_phase3_structure_analysis.py`; `src/stage7_phase4_text_evidence.py` | 否 |

## Phase 2 问题三周末共线

| 问题 | 改动 | 文件/脚本 | 是否影响数字 |
|---|---|---|---|
| 主回归同时包含 `is_weekend` 与 `C(weekday_str)`，周末与星期固定效应机械共线 | 主回归移除 `is_weekend`；周末仅由不含星期固定效应的补充模型估计；新增 `q3_weekend_supplement_coefficient.csv` | `src/stage4_q3_analysis.py`; `tables/q3_regression_coefficients_external.csv`; `tables/q3_weekend_supplement_coefficient.csv`; `outputs/stage4_q3_factor_analysis_report.md`; `paper/latex/sections/q3_model.tex`; `paper/latex/sections/assumptions_symbols.tex` | 是；表 11 周末主模型行由旧值 `0.041` 删除；补充模型周末系数为 `0.065067`、聚类标准误 `0.009329`、`p=3.0735e-12` |
| stage4 中间报告写“标准误按日期聚类”，但代码实际按门店-商品组合聚类 | 报告文字改为“标准误按门店-商品组合聚类” | `src/stage4_q3_analysis.py`; `outputs/stage4_q3_factor_analysis_report.md` | 否 |
| Q3 主要结论需保持 | 重新运行后核对 Q3 主模型 | `tables/q3_regression_model_summary.csv`; `tables/q3_regression_coefficients_external.csv` | 主结论未变：`N=58517`，调整 `R^2=0.370073`，活动日系数 `0.324632` |

## Phase 3 报告正式度

| 问题 | 改动 | 文件/脚本 | 是否影响数字 |
|---|---|---|---|
| `outputs/*.md` 存在问答式标题 | 将“什么是/为什么/如何”式标题改为正式报告标题，并重写口语化句子 | `src/stage3_q2_analysis.py`; `src/stage4_q3_analysis.py`; `src/stage5_q4_analysis.py`; `outputs/stage2_q1_modeling_report.md`; `outputs/stage3_q2_relationship_report.md`; `outputs/stage4_q3_factor_analysis_report.md`; `outputs/stage5_q4_final_model_report.md` | 否 |
| 电子附件说明不清 | README 与附录 E 明确 `outputs/*.md` 为内部过程记录，电子附件优先提交最终预测 CSV、关键表图和 `src/` | `README.md`; `paper/latex/sections/appendix.tex` | 否 |

## Phase 4 整洁度

| 问题 | 改动 | 文件/脚本 | 是否影响数字 |
|---|---|---|---|
| `METRICS` 写作 `MAPE`，与项目实际 WAPE 不一致 | 改为 `["MAE", "RMSE", "WAPE"]`；删除未被引用的历史 `LAG_DAYS`、`ROLLING_WINDOWS` 常量 | `src/config.py` | 否 |
| Q2 热力图函数存在未使用 `title` 参数；部分图未进论文 | 删除 `save_heatmap` 死参数；将未进论文图标注为探索性附图 | `src/stage3_q2_analysis.py` | 否 |
| 门店级零销量占比与门店-商品间歇需求口径易混 | 补充门店级零销量占比来自完整日期补齐面板，与问题四细粒度间歇需求口径不同 | `paper/latex/sections/q1_model.tex` | 否 |
| “经人工确认”缺少可核验依据 | 补充负销量记录数量合计、销售金额同步为负；补充 11001020 两个名称规格/单位/类别一致且价格区间重叠；补充温度列 760 条均满足前者不低于后者 | `paper/latex/sections/data_preprocessing.tex`; `paper/latex/sections/assumptions_symbols.tex`; `src/stage4_q3_analysis.py` | 否；新增说明数字来自原始附件核验，不改变模型结果 |

## Phase 5 终检

| 检查项 | 结果 |
|---|---|
| 完整 Python 管线 | 从仓库根按 README 顺序全部运行成功 |
| `stage3`/`stage4`/`stage5_q4_analysis` | 均无 `NameError`，summary JSON 正常写出 |
| `stage6`/`stage7` | 均可直接 `python src/...py` 运行 |
| Q1 关键数字 | 门店 `37.49%`；商品 `38.14%`；门店-商品 `74.95%` |
| Q2 关键数字 | 类别聚合后直接预测 WAPE `34.74%` |
| Q3 关键数字 | `N=58517`；调整 `R^2=0.370073`；活动日系数 `0.324632`；主模型无周末行 |
| Q4 关键数字 | 严格递推 SES `76.35%`；Ridge `76.68%`；混合策略 `75.85%` |
| 附录 D 汇总一致性 | 门店、商品、类别、门店-商品、逐日明细合计均为 `1328` |
| Q4 显著性 | 严格递推 `q4_significance_tests.csv` 结论均为“未达统计显著改进”；日滚动旁路文件为 `q4_significance_tests_daily.csv` |
| AI 小标题搜索 | `outputs/*.md`、论文和相关脚本中无“什么是/为什么/如何”式标题残留 |
| 占位符搜索 | 未发现 `TODO`、`FIXME`、`TBD`、`PLACEHOLDER`、`占位`、`待填`、`xxx` |
| LaTeX 编译 | `latexmk -xelatex` 成功生成 `paper/latex/main.pdf`；覆盖 `submission/final_paper.pdf` 时目标文件被外部进程占用，未能替换该副本 |
