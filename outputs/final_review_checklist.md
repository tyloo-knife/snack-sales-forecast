# 最终审查清单

生成时间：2026-05-03

## 1. 文件完整性

| 检查项 | 状态 | 备注 |
|---|---|---|
| 完整论文初稿 | 已生成 | `paper/full_paper_draft.md` |
| 模型评价章节 | 已补全 | `paper/model_evaluation.md` |
| 答辩材料 | 已生成 | `outputs/defense_notes.md` |
| AI 使用记录 | 已生成 | `outputs/ai_usage_record.md` |
| 最终预测表 | 已存在 | `outputs/final_7day_forecast.csv`，本次未覆盖 |
| 整数预测表 | 已生成 | `outputs/final_7day_forecast_integer.csv`，按件数提交时使用 |
| 最终方法选择 | 已存在 | `outputs/method_search/final_method_selection.md` |

## 2. 方法口径审查

| 检查项 | 结果 |
|---|---|
| 问题一是否保留移动平均、同星期均值、简单指数平滑 | 是 |
| 问题一是否写明门店主模型为移动平均、商品和门店-商品主模型为简单指数平滑 | 是 |
| 问题一是否补充 7 日窗口总量验证 | 是 |
| 问题二是否以附件 `category` 作为正式类别依据 | 是 |
| 问题二是否避免用相关性聚类替代主方案 | 是 |
| 问题三是否采用合并天气 + `log1p` 销量 + 历史控制固定效应回归为主分析 | 是 |
| 问题三是否避免因果化表述 | 是 |
| 问题四是否以严格 7 日递推为主验证口径 | 是 |
| 问题四是否说明综合 Ridge 未超过强 baseline | 是 |
| 是否保留 `outputs/final_7day_forecast.csv` 不覆盖 | 是 |
| 是否生成不覆盖原表的整数化预测副本 | 是 |
| 图表编号是否已在论文主稿中统一说明 | 是，表 1-6、图 1-16 |

## 3. 指标一致性审查

| 指标 | 论文写法 | 来源 |
|---|---:|---|
| 问题一门店主模型 WAPE | 37.49% | `tables/q1_model_metrics.csv` |
| 问题一商品主模型 WAPE | 38.14% | `tables/q1_model_metrics.csv` |
| 问题一门店-商品主模型 WAPE | 74.95% | `tables/q1_model_metrics.csv` |
| 问题一 7 日窗口商品 WAPE | 21.51% | `tables/method_search/q1_model_comparison.csv` |
| 问题二类别主模型 WAPE | 34.74% | `tables/q2_category_method_metrics.csv` |
| 问题三主回归 WAPE | 84.53% | `tables/method_search/q3_method_validation_metrics.csv` |
| 问题四严格递推强 baseline WAPE | 76.35% | `tables/q4_recursive_7day_model_metrics.csv` |
| 问题四严格递推综合 Ridge WAPE | 76.68% | `tables/q4_recursive_7day_model_metrics.csv` |
| 最终预测 7 天总销量，小数表 | 1311.294 | `outputs/final_7day_forecast.csv` |
| 最终预测 7 天总销量，整数表 | 1309 | `outputs/final_7day_forecast_integer.csv` |

## 4. 不能出现的表述

- 天气导致销量增加。
- 活动日导致销量显著增加。
- 节假日导致销量增加。
- 商品 A 带动商品 B 销售。
- 负相关商品就是替代品。
- 综合模型显著提高未来 7 天预测精度。
- 日滚动一步预测已经证明严格 7 日预测显著改进。
- 相关性聚类替代附件业务类别。

## 5. 提交前人工确认

1. 确认竞赛要求的预测表粒度：每日、7 天总量，或二者都要。
2. 确认正式提交应使用小数预测表还是整数预测表。
3. 如提交整数表，确认四舍五入规则是否符合赛题或组织方要求。
4. 检查所有图表编号是否与正式论文模板一致。
5. 检查参考文献格式是否满足学校或竞赛要求。
6. 检查 AI 使用记录是否需要随论文提交。
7. 检查是否需要把内部路径表述改成附录编号。
