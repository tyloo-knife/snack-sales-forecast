# 附录

## 附录 A：核心代码文件

- 数据读取与清洗：`src/data_loader.py`、`src/preprocessing.py`
- 特征工程：`src/features.py`
- 问题二商品关联与类别预测：`src/stage3_q2_analysis.py`
- 问题三外部因素分析：`src/stage4_q3_analysis.py`
- 问题四综合模型：`src/stage5_q4_analysis.py`
- 严格 7 日递推验证：`src/stage5_q4_recursive_validation.py`
- 问题四误差诊断：`src/stage5_q4_error_diagnosis.py`
- 天气敏感性检验：`src/stage5_q4_weather_sensitivity.py`
- 严格递推消融检验：`src/stage5_q4_ablation.py`
- 低销量鲁棒性策略：`src/stage5_q4_low_volume_strategy.py`
- 低销量混合最终预测：`src/stage5_q4_hybrid_final_forecast.py`
- 间歇性需求审查：`src/intermittent_demand_review.py`

## 附录 B：主要结果文件

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

若比赛提交要求销量为整数件数，优先使用混合策略整数预测表。整数化规则为：预测值小于 0 时置为 0，其余预测值四舍五入为整数；原小数预测表不覆盖，保留为模型直接输出依据。

## 附录 C：主要图表文件

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

## 附录 D：核心代码附录建议

正式 PDF 末尾已通过 LaTeX 附上以下核心代码：

- `src/stage5_q4_ablation.py`
- `src/stage5_q4_low_volume_strategy.py`
- `src/stage5_q4_hybrid_final_forecast.py`
