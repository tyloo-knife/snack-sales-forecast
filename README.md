# 休闲零食连锁店商品销量预测

## 1. 项目简介

本项目对应数学建模 A 题《休闲零食连锁店商品销量预测》，目标是基于历史销售明细和天气、节假日、活动日数据，预测未来 7 天休闲零食销量。
项目已完成数据读取、预处理、四问建模、严格 7 日递推验证、消融检验、低销量鲁棒性检验、候选优化复核、预测不确定性补充和论文终稿排版材料。
当前 README 是项目导航页和复现入口，不是论文正文；详细模型推导、图表解释和结论见各阶段报告与 `paper/`。

## 2. 当前进度

| 阶段 | 内容 | 状态 | 主要产物 |
|---|---|---|---|
| 阶段 0 | 题目理解与总体方案 | 已完成 | `outputs/stage0_problem_understanding.md` |
| 阶段 1 | 数据读取与预处理 | 已完成 | `outputs/stage1_data_inspection_report.md`、`data/processed/` |
| 阶段 2 | 问题一建模 | 已完成 | `outputs/stage2_q1_modeling_report.md`、`tables/q1_*.csv`、`figures/q1_*.png` |
| 阶段 3 | 问题二建模 | 已完成 | `outputs/stage3_q2_relationship_report.md`、`tables/q2_*.csv`、`figures/q2_*.png` |
| 阶段 4 | 问题三建模 | 已完成 | `outputs/stage4_q3_factor_analysis_report.md`、`outputs/q3_causality_language_review.md` |
| 阶段 5 | 问题四综合模型 | 已完成，已补充信息泄露检查、严格递推、消融、低销量策略、候选优化复核和预测区间 | `outputs/stage5_q4_final_model_report.md`、`outputs/q4_recursive_7day_model_comparison.md`、`outputs/q4_ablation_report.md`、`outputs/q4_low_volume_strategy_report.md`、`outputs/q4_candidate_model_review.md`、`outputs/forecast_uncertainty_report.md` |
| 阶段 6 | 完整论文写作 | 已完成并进入最终格式迭代 | `paper/full_paper_draft.md`、`paper/latex/`、`paper/final_paper.pdf` |
| 阶段 7 | 答辩与最终检查 | 已完成并更新 | `outputs/final_review_checklist.md`、`outputs/defense_notes.md`、`outputs/final_submission_review.md` |

## 3. 目录结构

| 路径 | 用途 |
|---|---|
| `data/raw/` | 题目文档和原始附件，只读，不覆盖 |
| `data/processed/` | 清洗后的日销量面板和建模基础表 |
| `notebooks/` | 分阶段数据检查、建模和预测 Notebook |
| `src/` | 数据读取、预处理、特征、模型、评价和阶段脚本 |
| `figures/` | 问题一至问题四生成的图表 |
| `tables/` | 各阶段统计表、验证结果、预测结果和误差比较表 |
| `outputs/` | 阶段报告、关键预测结果、中间结论和专项检查 |
| `paper/` | 后续论文写作使用的分章节 Markdown 材料 |
| `archive/` | 旧版本或归档材料 |

## 4. 数据说明

- 原始数据位置：`data/raw/`。
- 已读取原始文件：`A-休闲零食连锁店商品销量预测.docx`、`附件一：历史零售明细统计.xlsx`、`附件二：某市近两年天气等数据.xlsx`。
- 处理后核心表：`data/processed/daily_store_product_sales.csv` 和 `data/processed/modeling_base_table.csv`。
- 字段说明位置：`DATA_DICTIONARY.md`，人工确认和待谨慎处理事项也记录在该文件中。
- 原始数据只读：不得修改或覆盖 `data/raw/` 中的文件。

## 5. 建模路线概述

### 问题一：门店与商品销量分析及未来 7 天预测

- 分析门店、商品、门店-商品组合的销量差异和星期效应。
- 使用移动平均、同星期均值、简单指数平滑作为短期预测模型。
- 主要产物：`outputs/stage2_q1_modeling_report.md`、`tables/q1_model_metrics.csv`、`tables/q1_*forecast*.csv`、`figures/q1_*.png`。

### 问题二：商品关联分析与类别聚合预测

- 构造日期 × 商品销量矩阵，计算商品销量相关系数。
- 按商品类别聚合，并比较“单品预测后加总”和“类别聚合后直接预测”。
- 主要产物：`outputs/stage3_q2_relationship_report.md`、`tables/q2_correlation_matrix_*.csv`、`tables/q2_category_method_metrics.csv`、`figures/q2_*.png`。

### 问题三：天气、节假日、活动日因素分析

- 将天气、最高温、最低温、风力、节假日、周末、活动日等变量纳入分析。
- 使用描述性统计、控制变量回归和随机森林置换重要性分析统计关联。
- 避免把统计关联写成因果结论；因果措辞审查见 `outputs/q3_causality_language_review.md`。
- 主要产物：`outputs/stage4_q3_factor_analysis_report.md`、`tables/q3_*.csv`、`figures/q3_*.png`。

### 问题四：综合预测模型与误差比较

- 综合使用历史销量滞后项、滚动均值、日期变量、门店、商品、类别、天气、节假日和活动日。
- 比较 baseline、Ridge、RandomForest，并与问题一、问题二模型做误差对比。
- 严格 7 日递推下，低销量指数平滑 + 常规序列 Ridge 混合策略 WAPE=75.85%，作为候选最终预测。
- 候选最终预测结果：`outputs/final_7day_forecast_hybrid_low_volume.csv`；原完整 Ridge 预测保留为 `outputs/final_7day_forecast.csv`；信息泄露检查：`outputs/q4_information_leakage_review.md`。

## 6. 主要模型与指标

| 类型 | 方法 | 用途 | 对应阶段 |
|---|---|---|---|
| 描述性统计 | 分组汇总、趋势图、星期效应 | 了解销量结构和数据问题 | 阶段 1、2、3、4 |
| Baseline | 移动平均、同星期均值 | 提供可解释的短期预测基准 | 阶段 2、3、5 |
| 时间序列平滑 | 简单指数平滑 | 预测门店、商品、类别销量 | 阶段 2、3 |
| 商品关联 | Pearson 相关系数矩阵 | 分析同门店不同商品销量联系 | 阶段 3 |
| 回归分析 | 多元线性回归与控制变量 | 分析外部因素统计关联 | 阶段 4 |
| 机器学习 | 随机森林置换重要性 | 辅助分析变量重要性 | 阶段 4、5 |
| 综合预测 | Ridge 回归、RandomForest | 门店-商品粒度综合预测 | 阶段 5 |
| 稳健混合 | 低销量指数平滑 + 常规序列 Ridge | 问题四最终候选预测 | 阶段 5 最终稳健性迭代 |

评价指标：

- MAE：平均绝对误差，表示平均预测偏差大小。
- RMSE：均方根误差，对较大的预测错误更敏感。
- WAPE：加权绝对百分比误差，适合存在许多低销量或零销量记录的销售预测。

## 7. 如何复现

1. Python 版本建议：Python 3.10+；本项目已使用 `C:\ProgramData\anaconda3\python.exe` 运行过阶段脚本。
2. 主要依赖：`pandas`、`numpy`、`matplotlib`、`scikit-learn`、`scipy`、`statsmodels`、`openpyxl`、`python-docx`、`jupyter`。
3. 运行前确认 `data/raw/` 中存在题目文档、附件一和附件二。
4. 推荐 Notebook 顺序：`00_environment_check.ipynb`、`01_data_inspection.ipynb`、`02_data_cleaning.ipynb`、`03_exploratory_analysis.ipynb`、`04_baseline_forecast.ipynb`、`05_product_relationship_and_category_forecast.ipynb`、`06_external_factor_analysis.ipynb`、`07_final_prediction.ipynb`。
5. 阶段脚本入口：`src/stage3_q2_analysis.py`、`src/stage4_q3_analysis.py`、`src/stage5_q4_analysis.py`、`src/stage5_q4_recursive_validation.py`、`src/stage5_q4_ablation.py`、`src/stage5_q4_low_volume_strategy.py`、`src/stage5_q4_hybrid_final_forecast.py`、`src/stage5_q4_award_iteration.py`。
6. 复现输出位置：图表写入 `figures/`，表格写入 `tables/`，阶段报告和关键预测表写入 `outputs/`。

## 8. 主要输出文件

| 文件 | 说明 |
|---|---|
| `outputs/stage0_problem_understanding.md` | 阶段 0 题目理解与总体方案 |
| `outputs/stage1_data_inspection_report.md` | 阶段 1 数据检查报告 |
| `outputs/stage1_preprocessing_summary.md` | 阶段 1 预处理摘要 |
| `outputs/stage2_q1_modeling_report.md` | 问题一建模报告 |
| `outputs/stage3_q2_relationship_report.md` | 问题二商品关联与类别预测报告 |
| `outputs/stage4_q3_factor_analysis_report.md` | 问题三外部因素统计关联分析报告 |
| `outputs/stage5_q4_final_model_report.md` | 问题四综合预测模型报告 |
| `outputs/q4_information_leakage_review.md` | 问题四信息泄露专项检查 |
| `outputs/q4_recursive_7day_model_comparison.md` | 问题四严格 7 日递推验证 |
| `outputs/q4_ablation_report.md` | 问题四严格递推消融检验 |
| `outputs/q4_low_volume_strategy_report.md` | 低销量鲁棒性策略检验 |
| `outputs/q4_hybrid_final_forecast_report.md` | 混合策略最终预测说明 |
| `outputs/q4_candidate_model_review.md` | 低销量阈值、兜底方法和层级校准候选复核 |
| `outputs/forecast_uncertainty_report.md` | 基于严格递推残差的预测不确定性报告 |
| `outputs/q4_error_attribution_enhanced_report.md` | 按 horizon、需求类别和日历场景增强误差归因 |
| `outputs/final_7day_forecast_hybrid_low_volume.csv` | 候选门店-商品-日期粒度最终 7 天预测结果 |
| `outputs/final_7day_forecast.csv` | 原完整 Ridge 门店-商品-日期粒度 7 天预测结果，对照保留 |
| `tables/q4_comparison_with_previous_models.csv` | 综合模型与问题一、二模型误差比较 |
| `tables/q4_significance_tests.csv` | 问题四显著性检验结果 |
| `figures/` | 各阶段图表目录 |
| `tables/` | 各阶段统计、验证和预测表目录 |
| `paper/` | 论文分章节 Markdown 材料 |
| `RESULT_LOG.md` | 实验记录和阶段推进日志 |

## 9. 论文写作入口

论文材料位于 `paper/`，当前包含完整 Markdown 草稿和 LaTeX 排版工程。LaTeX 已按“第一页论文题目和成员信息、第二页摘要、第三页起正文、末尾附代码”的格式调整。
论文写作时应重点引用 `stage2` 至 `stage5` 报告、`DATA_DICTIONARY.md`、`ASSUMPTIONS.md` 和 `REVIEW_CHECKLIST.md`。
README 不写论文正文，只作为索引和复现入口。

## 10. 下一步工作

- LaTeX 封面成员真实信息已填写；正式提交前核对是否与报名系统一致。
- 如需修改封面，可编辑 `paper/cover_member_info.csv` 后运行 `C:\ProgramData\anaconda3\python.exe src\fill_cover_members.py`；若匿名提交，运行 `C:\ProgramData\anaconda3\python.exe src\fill_cover_members.py --anonymous`。
- 已编译并同步 `paper/final_paper.pdf` 与 `submission/final_paper.pdf`，正式上传前重点复核封面、摘要页、图表和附录代码。
- 正式提交前确认使用混合策略预测表还是原完整 Ridge 对照表，并确认小数/整数口径。
- 检查 AI 使用披露是否符合校赛要求。

## 11. 注意事项

- `data/raw/` 原始数据不得覆盖或修改。
- 所有结论必须来自附件数据、处理后数据或明确模型计算。
- 天气、节假日、活动日分析只能表述为统计关联，不能直接写成因果结论。
- 时间序列预测不能随机打乱，应使用时间切分、滚动验证或递推验证。
- 最终论文不得伪造指标、图表、字段或预测结果。
- 问题四主结论必须以严格 7 日递推验证为准，日滚动一步预测只作辅助。
- AI 辅助内容需要按竞赛要求记录，论文中应保留可复现代码和结果依据。
