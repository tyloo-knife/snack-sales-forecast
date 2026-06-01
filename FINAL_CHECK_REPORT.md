# 最终复现与一致性终检报告

日期：2026-06-01

## 1. 全流程脚本退出码

本轮使用全新临时虚拟环境 `C:\Users\20215\AppData\Local\Temp\snack-sales-forecast-phase0-20260601161354`，执行 `pip install -r requirements.txt` 成功，未发现缺包，未修改 `requirements.txt`。

| 顺序 | 脚本 | 退出码 | traceback |
|---:|---|---:|---|
| 1 | `src/stage2_q1_figures.py` | 0 | 无 |
| 2 | `src/stage6_phase3_structure_analysis.py` | 0 | 无 |
| 3 | `src/stage7_phase4_text_evidence.py` | 0 | 无 |
| 4 | `src/stage3_q2_analysis.py` | 0 | 无 |
| 5 | `src/stage4_q3_analysis.py` | 0 | 无 |
| 6 | `src/stage5_q4_analysis.py` | 0 | 无 |
| 7 | `src/stage5_q4_recursive_validation.py` | 0 | 无 |
| 8 | `src/stage5_q4_ablation.py` | 0 | 无 |
| 9 | `src/stage5_q4_low_volume_strategy.py` | 0 | 无 |
| 10 | `src/stage5_q4_hybrid_final_forecast.py` | 0 | 无 |
| 11 | `src/stage5_q4_error_diagnosis.py` | 0 | 无 |
| 12 | `src/stage5_q4_weather_sensitivity.py` | 0 | 无 |
| 13 | `src/intermittent_demand_review.py` | 0 | 无 |

`outputs/stage3_q2_summary.json`、`outputs/stage4_q3_summary.json`、`outputs/stage5_q4_summary.json` 均已成功写出。

## 2. 论文编译结果

使用 `latexmk -g -xelatex -interaction=nonstopmode main.tex` 在 `paper/latex/` 下强制重建，退出码为 0，并已同步 `submission/final_paper.pdf`。

| 检查项 | 结果 |
|---|---|
| undefined references | 0 |
| undefined citations | 0 |
| 缺图错误 | 0 |
| LaTeX Error | 0 |
| overfull hbox/vbox | 0 |

编译日志仅有 underfull hbox 与 `xdvipdfmx` 页对象重复定义 warning，不构成编译错误或 overfull 排版错位。

## 3. 文件引用审计表

| 类别 | 结果 |
|---|---|
| 论文提到但不存在的文件 | 无 |
| `tables/q3_regression_coefficients_external.csv` | 存在，已被附录 E 引用 |
| `tables/q3_weekend_supplement_coefficient.csv` | 存在，已被附录 E 引用 |
| `tables/q4_significance_tests.csv` | 存在，已被附录 E 引用 |
| `tables/q4_significance_tests_daily.csv` | 存在，已被附录 E 引用 |
| 最终整数预测明细 | `outputs/final_7day_forecast_hybrid_low_volume_integer.csv` 存在，已被 README 与附录 E 引用 |
| 连续预测表 | `outputs/final_7day_forecast_hybrid_low_volume.csv` 存在，已被 README 与附录 E 引用 |
| 关键产物存在但论文漏提 | 必核关键产物无漏提；其余阶段性 CSV 由“阶段报告、误差表、图表和最终预测表”总括说明 |

## 4. 数字与源 CSV 审计表

| 表 | 源 CSV | 结论 |
|---|---|---|
| 表11 问题三固定效应回归主要系数 | `tables/q3_regression_coefficients_external.csv`、`tables/q3_regression_model_summary.csv` | 一致：10 行系数、标准误、t 值、p 值、N=58517、调整 R2=0.370 均匹配 |
| 表16 严格递推误差显著性检验 | `tables/q4_significance_tests.csv` | 一致：两行 WAPE、WAPE 差、Wilcoxon p、bootstrap 95% CI 均匹配 |
| 表19 alpha 灵敏度 | `tables/q1_exp_smoothing_alpha_sensitivity.csv` | 一致：3 个层级 × 3 个 alpha 的 WAPE 均匹配 |
| 表25 门店预测总量 | `tables/q4_hybrid_forecast_7day_total_by_store.csv` | 一致：7 个门店逐项匹配，总量 1328 |
| 表27 类别预测总量 | `tables/q4_hybrid_forecast_7day_total_by_category.csv` | 一致：8 个类别逐项匹配，总量 1328 |
| 表29 商品预测总量 | `tables/q4_hybrid_forecast_7day_total_by_product.csv` | 一致：12 个商品逐项匹配，总量 1328 |

复现后部分 CSV 相对仓库基线出现浮点尾差，最大绝对差为 `9.095e-13`，均集中在随机森林相关预测、误差或重要性小数末位；无非数值字段变化，论文展示精度下不产生可见数字变化。

## 5. 口径一致性结论

| 项目 | 结论 |
|---|---|
| 层级汇总总量 | 全文与说明文件只保留 1328；未检出 1330 或 1331 |
| 问题三主回归 | `tables/q3_regression_coefficients_external.csv` 为 10 行，且无周末行 |
| 周末处理 | 周末仅在 `tables/q3_weekend_supplement_coefficient.csv` 的补充模型中估计，系数为 0.065；正文已明确主模型不单独识别周末效应 |
| WAPE 口径 | 数值型 WAPE 出现处均由同句、表题或前文限定验证口径；未来预测汇总明确不是误差评价表，不计算 WAPE |
| 结论表述 | 保留“综合 Ridge 未跑赢问题一强基准”“低销量混合策略未达显著改进”“外部因素只解释为统计关联”“未来 7 天为条件预测”等诚实结论 |

## 6. 整洁化清单

| 文件 | 处理 |
|---|---|
| `src/config.py` | 删除未被任何脚本导入的旧常量：`OUTPUT_DIR`、`FIGURE_DIR`、`TABLE_DIR`、`RANDOM_STATE`、`TRAIN_RATIO`、`METRICS` |
| `src/visualization.py` | 删除未被当前 README 复现入口和 `src` 内模块调用的旧 Pillow 绘图模块 |
| `README.md` | 明确列出 `_daily`、`_weekend_supplement`、最终整数预测明细和连续预测表 |
| `paper/latex/sections/appendix.tex` | 将 stage6 与 stage7 拆为两个条目，脚本清单与 README 保持 13 项、顺序完全一致 |
| `paper/latex/sections/q3_model.tex` | 强化问题三周末口径：周末仅用于描述统计和不含星期固定效应的补充模型 |
| `figures/` | 未删除图。所有 `\includegraphics` 文件存在且编译通过；未判定出安全可删的废图 |

未改动参赛人员信息页。

## 7. 电子附件清单建议

建议纳入：

| 类别 | 文件或目录 |
|---|---|
| 最终论文 | `submission/final_paper.pdf` |
| 最终整数预测明细 | `outputs/final_7day_forecast_hybrid_low_volume_integer.csv`；提交副本为 `submission/final_7day_forecast_integer.csv` |
| 最终连续预测表 | `outputs/final_7day_forecast_hybrid_low_volume.csv`；提交副本为 `submission/final_7day_forecast.csv` |
| 关键结果表 | `tables/q1_model_metrics.csv`、`tables/q2_category_method_metrics.csv`、`tables/q3_regression_coefficients_external.csv`、`tables/q3_weekend_supplement_coefficient.csv`、`tables/q4_significance_tests.csv`、`tables/q4_significance_tests_daily.csv`、`tables/q4_ablation_metrics.csv`、`tables/q4_low_volume_strategy_metrics.csv`、`tables/q4_weather_sensitivity_metrics.csv`、`tables/q4_hybrid_forecast_7day_total_by_store.csv`、`tables/q4_hybrid_forecast_7day_total_by_product.csv`、`tables/q4_hybrid_forecast_7day_total_by_category.csv`、`tables/q4_hybrid_forecast_7day_total_by_store_product.csv` |
| 关键图 | `figures/technical_route_phase3.png`、`figures/q1_daily_total_trend.png`、`figures/q2_overall_product_correlation_heatmap.png`、`figures/q3_regression_factor_effect_ranking.png`、`figures/q4_hybrid_forecast_7day_total_by_store.png`、`figures/q4_hybrid_forecast_7day_total_by_category.png`、`figures/q4_actual_vs_predicted_scatter.png` |
| 代码 | `src/` 全部当前代码 |
| 说明文件 | `README.md`、`DATA_DICTIONARY.md`、`requirements.txt` |
| 可选说明 | `submission/README_submission.md`、`submission/ai_usage_record.md`，按课程或竞赛提交规则处理 |

不建议纳入：`outputs/*.md`。这些文件定位为内部过程记录和复现说明，附录 E 已说明其性质；正式提交优先采用论文、预测 CSV、关键结果表图、代码和数据字典。

## 8. 个人信息终检

LaTeX 源文件中个人信息仅出现在 `paper/latex/main.tex` 的 `titlepage` 成员信息表，对应指定封面页；`paper/latex/sections/` 正文未检出姓名、学号或电话。`paper/cover_member_info_template.md` 为封面成员信息模板，保持原样。
