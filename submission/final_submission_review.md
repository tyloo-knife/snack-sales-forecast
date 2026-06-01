# 提交前最终审查报告

生成时间：2026-05-31

## 1. 当前项目状态

项目已完成阶段 0 至阶段 7 的主要材料整理。本轮迭代未修改 `data/raw/`，未覆盖 `outputs/final_7day_forecast.csv`，新增的是严格递推消融、低销量鲁棒性检验、不覆盖原结果的混合策略预测表、候选优化复核、误差归因增强和经验预测区间。审查范围集中在论文结构、图表编号、问题三因果表述、问题四验证口径、预测表整数化副本、答辩材料、提交包口径和 AI 使用记录。

## 2. 已完成材料清单

| 材料 | 状态 | 路径 |
|---|---|---|
| 完整论文初稿 | 已完成 | `paper/full_paper_draft.md` |
| 正式论文 PDF | 已编译同步 | `paper/final_paper.pdf`、`submission/final_paper.pdf` |
| 封面成员信息 | 已填写 | `paper/latex/main.tex`、`paper/cover_member_info.csv` |
| 模型评价章节 | 已补全 | `paper/model_evaluation.md` |
| 答辩准备材料 | 已补充 | `outputs/defense_notes.md` |
| AI 使用记录 | 已补充 | `outputs/ai_usage_record.md` |
| 最终审查清单 | 已更新 | `outputs/final_review_checklist.md` |
| 小数预测表 | 已保留 | `outputs/final_7day_forecast.csv` |
| 整数预测表 | 已生成 | `outputs/final_7day_forecast_integer.csv` |
| 混合策略预测表 | 已生成 | `outputs/final_7day_forecast_hybrid_low_volume.csv` |
| 混合策略整数表 | 已生成 | `outputs/final_7day_forecast_hybrid_low_volume_integer.csv` |
| 方法最终选择 | 已完成 | `outputs/method_search/final_method_selection.md` |
| 问题四信息泄露审查 | 已完成 | `outputs/q4_information_leakage_review.md` |
| 严格 7 日递推验证 | 已完成 | `outputs/q4_recursive_7day_model_comparison.md` |
| 严格递推消融检验 | 已完成 | `outputs/q4_ablation_report.md` |
| 低销量鲁棒性检验 | 已完成 | `outputs/q4_low_volume_strategy_report.md` |
| 混合策略最终预测说明 | 已完成 | `outputs/q4_hybrid_final_forecast_report.md` |
| 候选优化复核 | 已完成 | `outputs/q4_candidate_model_review.md` |
| 预测不确定性报告 | 已完成 | `outputs/forecast_uncertainty_report.md` |
| 增强误差归因 | 已完成 | `outputs/q4_error_attribution_enhanced_report.md` |
| 误差诊断 | 已完成 | `outputs/q4_error_diagnosis_report.md` |
| 天气敏感性检验 | 已完成 | `outputs/q4_weather_sensitivity_report.md` |
| 间歇性需求审查 | 已完成 | `outputs/intermittent_demand_review.md` |

## 3. 论文主结论

1. 问题一保留移动平均、同星期均值和简单指数平滑。门店层面以移动平均为主，商品和门店-商品层面以简单指数平滑为主。
2. 问题二以附件 `category` 字段作为同类零食整合依据，类别聚合后直接预测 + 简单指数平滑为主方案。
3. 问题三采用合并天气、`log1p` 销量和历史控制固定效应回归分析外部因素与销量之间的条件统计关联，不作因果解释。
4. 问题四采用综合 Ridge 模型融合历史销量、类别、日历和外部变量，但严格 7 日递推主口径下未超过问题一门店-商品简单指数平滑强 baseline；进一步采用低销量指数平滑、常规序列 Ridge 的混合策略，严格递推 WAPE=75.85%，作为候选最终预测方案。近 28 日均值兜底探索候选在同一验证集上 WAPE=74.95%，但留一窗口选模平均改善只有 0.19 个百分点，未替换当前混合策略。

## 4. 问题四验证口径说明

日滚动一步预测只作为辅助口径，适合解释“每天更新真实销量后预测下一天”。严格 7 日递推验证是主口径，适合解释“窗口开始时一次性预测未来 7 天”。论文中已统一写明：严格递推下问题一门店-商品简单指数平滑 WAPE=76.35%，综合 Ridge WAPE=76.68%，综合 Ridge 未超过强 baseline，不能写成“显著优于所有模型”。

未来天气、温度、风力和活动日没有附件真实观测，最终预测使用历史同期参考情景。因此 `outputs/final_7day_forecast_hybrid_low_volume.csv` 和原 `outputs/final_7day_forecast.csv` 都是条件预测表，不是真实未来外部变量观测下的确定结果。

## 5. 预测表整数化

已生成混合策略整数化表 `outputs/final_7day_forecast_hybrid_low_volume_integer.csv`，同时保留原完整 Ridge 整数化表 `outputs/final_7day_forecast_integer.csv`。整数化规则为：

1. 预测值小于 0 的置为 0；
2. 其余预测值四舍五入为整数；
3. 不覆盖原始小数预测表。
4. 极小预测值低于 `1e-6` 时置为 0，避免提交表出现接近 0 的科学计数法残留。

混合策略小数预测表 7 天预测总销量为 1329.894，整数预测表 7 天预测总销量为 1328。原完整 Ridge 小数预测表 7 天总量为 1311.294，整数预测表 7 天总量为 1309。正式提交前需确认比赛要求使用小数预测还是整数件数，并确认最终提交采用混合策略还是原完整 Ridge 对照。

## 6. 尚存风险

1. 论文仍需按学校或竞赛模板进行最终排版，当前 Markdown 版本不等于最终排版稿。
2. 图表文件路径已核查存在，但正式排版时还需检查图片清晰度、尺寸和标题位置。
3. 问题三外部因素分析仍是观察数据下的统计关联，不能扩展为因果结论。
4. 问题四严格递推验证只有 4 个完整 7 日窗口，显著性检验和稳定性判断应谨慎。
5. 低销量、零销量和间歇性需求使门店-商品粒度 WAPE 较高，答辩中需要主动说明。
6. 最终预测依赖历史同期天气和活动情景，若实际未来天气或活动安排不同，预测会变化。
7. 当前 `submission/` 目录已同步为混合策略口径，正式提交前仅需人工确认竞赛平台要求的文件命名、预测粒度和小数/整数格式。

## 7. 提交前必须人工确认事项

1. 正式论文是否采用 `paper/full_paper_draft.md` 为唯一主稿，还是需要拆分章节合并到学校模板。
2. 预测表提交粒度是每日门店-商品、7 天汇总，还是需要同时提交多个层级。
3. 正式提交使用混合策略预测表还是原完整 Ridge 对照表，并确认小数/整数口径。
4. AI 使用记录是否需要单独随论文提交，措辞是否符合竞赛规则。
5. 参考文献格式、图表编号和附录代码是否符合最终排版要求。
6. 是否需要移除提交包中的中间探索报告，仅保留论文、最终预测表、关键图表、代码和 AI 使用记录。
