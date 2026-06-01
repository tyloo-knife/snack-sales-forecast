# 最终审查清单

生成时间：2026-05-31

## 1. 文件完整性

| 检查项 | 状态 | 备注 |
|---|---|---|
| 完整论文初稿 | 已生成 | `paper/full_paper_draft.md` |
| 模型评价章节 | 已补全 | `paper/model_evaluation.md` |
| 答辩材料 | 已生成 | `outputs/defense_notes.md` |
| AI 使用记录 | 已生成 | `outputs/ai_usage_record.md` |
| 最终预测表 | 已存在 | `outputs/final_7day_forecast.csv`，本次未覆盖 |
| 整数预测表 | 已生成 | `outputs/final_7day_forecast_integer.csv`，按件数提交时使用 |
| 混合策略预测表 | 已生成 | `outputs/final_7day_forecast_hybrid_low_volume.csv`，候选最终预测 |
| 混合策略整数表 | 已生成 | `outputs/final_7day_forecast_hybrid_low_volume_integer.csv` |
| 候选优化复核 | 已生成 | `outputs/q4_candidate_model_review.md` |
| 预测不确定性报告 | 已生成 | `outputs/forecast_uncertainty_report.md` |
| 增强误差归因 | 已生成 | `outputs/q4_error_attribution_enhanced_report.md` |
| 最终冻结审查 | 已生成 | `outputs/final_freeze_audit_20260531.md` |
| 封面成员信息模板 | 已生成 | `paper/cover_member_info_template.md` |
| 封面成员信息 CSV | 已填写 | `paper/cover_member_info.csv` |
| 封面填充脚本 | 已生成 | `src/fill_cover_members.py` |
| 正式封面成员信息 | 已写入 PDF | `paper/latex/main.tex`、`paper/final_paper.pdf`、`submission/final_paper.pdf` |
| 最终方法选择 | 已存在 | `outputs/method_search/final_method_selection.md` |
| 严格递推消融检验 | 已生成 | `outputs/q4_ablation_report.md` |
| 低销量鲁棒性检验 | 已生成 | `outputs/q4_low_volume_strategy_report.md` |

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
| 是否补充严格递推消融检验 | 是 |
| 是否补充低销量鲁棒性混合策略 | 是，WAPE=75.85% |
| 是否补充低销量阈值与兜底候选复核 | 是；全样本最佳 WAPE=74.95%，但留一窗口改善 0.19 个百分点，未替换主方案 |
| 是否补充预测区间/不确定性 | 是；基于严格递推 7 日残差经验区间 |
| 是否保留 `outputs/final_7day_forecast.csv` 不覆盖 | 是 |
| 是否生成不覆盖原表的整数化预测副本 | 是 |
| 图表编号是否已在论文主稿中统一说明 | 是，Q4 预测图已改为混合策略图，新增不确定性区间图 |

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
| 问题四低销量混合策略 WAPE | 75.85% | `tables/q4_low_volume_strategy_metrics.csv` |
| 近 28 日均值兜底探索候选 WAPE | 74.95% | `tables/q4_candidate_model_metrics.csv`，不替换主方案 |
| 留一窗口选模平均改善 | 0.19 个百分点 | `tables/q4_candidate_leave_one_window_selection.csv` |
| 当前混合策略低销量序列整体 WAPE | 122.53% | `tables/q4_hybrid_error_by_demand_class.csv` |
| 最终预测 7 天总销量，小数表 | 1311.294 | `outputs/final_7day_forecast.csv` |
| 最终预测 7 天总销量，整数表 | 1309 | `outputs/final_7day_forecast_integer.csv` |
| 混合策略最终预测 7 天总销量，小数表 | 1329.894 | `outputs/final_7day_forecast_hybrid_low_volume.csv` |
| 混合策略最终预测 7 天总销量，整数表 | 1328 | `outputs/final_7day_forecast_hybrid_low_volume_integer.csv` |

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
3. 确认正式提交采用混合策略预测表还是原完整 Ridge 对照表。
4. 如提交整数表，确认四舍五入规则是否符合赛题或组织方要求。
5. 检查所有图表编号是否与正式论文模板一致。
6. 检查参考文献格式是否满足学校或竞赛要求。
7. 检查 AI 使用记录是否需要随论文提交。
8. 检查是否需要把内部路径表述改成附录编号。
9. 复核封面成员信息是否与报名系统一致；若竞赛匿名提交，按匿名规则重新生成 PDF。
