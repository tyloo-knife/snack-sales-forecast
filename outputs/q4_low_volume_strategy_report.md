# 问题四低销量序列鲁棒性策略检验报告

执行日期：2026-06-01

## 1. 检验目的

门店-商品序列中低销量和零销量较多，最细粒度 WAPE 偏高。为避免盲目引入 Croston/TSB 等复杂间歇需求模型，本检验只评估可解释的低销量分层后处理策略：每个验证窗口开始日前，根据历史平均销量、零销量比例和非零销量比例识别低销量序列；低销量序列可向“近 28 日均值、商品历史均值、类别历史均值”的层级均值收缩。

分类和收缩目标均只使用窗口开始日前历史数据，未使用验证窗口真实销量。

## 2. 低销量分类概况

| demand_class | is_low_volume | mean_series_count | mean_history_sales | mean_zero_ratio |
| --- | --- | --- | --- | --- |
| low_volume_intermittent | 1 | 10.750 | 0.508 | 0.773 |
| no_positive_history | 1 | 1.000 | 0.000 | 1.000 |
| very_sparse | 1 | 18.500 | 0.109 | 0.938 |
| intermittent | 0 | 24.000 | 1.727 | 0.629 |
| regular | 0 | 12.000 | 3.041 | 0.302 |
| volatile | 0 | 10.000 | 4.261 | 0.410 |

逐窗口分类明细见 `tables/q4_low_volume_classification.csv`。

## 3. 门店-商品主粒度策略比较

| model_label | strategy_rule | MAE | RMSE | WAPE_pct | WAPE_pct_point_change_vs_q1 | WAPE_pct_point_change_vs_ridge | actual_sum | prediction_sum | bias_pct_of_actual | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 低销量指数平滑-常规Ridge混合 | 低销量序列使用指数平滑，非低销量序列使用完整 Ridge | 1.804 | 3.718 | 75.853 | -0.494 | -0.831 | 5061.000 | 4939.716 | -0.024 | 2128 |
| 问题一简单指数平滑 | 原问题一门店-商品指数平滑，作为严格递推强基准 | 1.816 | 3.772 | 76.346 | 0.000 | -0.338 | 5061.000 | 4732.862 | -0.065 | 2128 |
| 低销量收缩-常规Ridge混合 | 低销量序列使用收缩指数平滑，非低销量序列使用完整 Ridge | 1.823 | 3.709 | 76.659 | 0.313 | -0.025 | 5061.000 | 4996.783 | -0.013 | 2128 |
| 完整综合Ridge | 原问题四完整综合 Ridge，作为综合模型基准 | 1.824 | 3.702 | 76.684 | 0.338 | 0.000 | 5061.000 | 5018.373 | -0.008 | 2128 |
| 低销量收缩指数平滑 | 低销量序列用 50% 指数平滑 + 50% 历史层级均值收缩，其余保持指数平滑 | 1.835 | 3.764 | 77.153 | 0.807 | 0.469 | 5061.000 | 4789.928 | -0.054 | 2128 |
| 低销量收缩Ridge | 低销量序列用 50% Ridge + 50% 历史层级均值收缩，其余保持 Ridge | 1.838 | 3.705 | 77.271 | 0.925 | 0.587 | 5061.000 | 5036.111 | -0.005 | 2128 |

完整分层指标见 `tables/q4_low_volume_strategy_metrics.csv`，按需求类别拆分指标见 `tables/q4_low_volume_strategy_by_class.csv`。图表已保存至 `figures/q4_low_volume_strategy_wape.png`。

## 4. 低销量类别误差

| model_label | demand_class | MAE | RMSE | WAPE_pct | actual_sum | prediction_sum | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 低销量收缩-常规Ridge混合 | low_volume_intermittent | 1.092 | 2.306 | 94.429 | 348.000 | 283.677 | 301 |
| 低销量收缩指数平滑 | low_volume_intermittent | 1.092 | 2.306 | 94.429 | 348.000 | 283.677 | 301 |
| 低销量收缩Ridge | low_volume_intermittent | 1.130 | 2.265 | 97.709 | 348.000 | 298.855 | 301 |
| 低销量指数平滑-常规Ridge混合 | low_volume_intermittent | 1.157 | 2.358 | 100.069 | 348.000 | 328.197 | 301 |
| 问题一简单指数平滑 | low_volume_intermittent | 1.157 | 2.358 | 100.069 | 348.000 | 328.197 | 301 |
| 完整综合Ridge | low_volume_intermittent | 1.193 | 2.251 | 103.171 | 348.000 | 358.551 | 301 |
| 低销量指数平滑-常规Ridge混合 | no_positive_history | 0.048 | 0.218 | 100.000 | 1.000 | 0.000 | 21 |
| 问题一简单指数平滑 | no_positive_history | 0.048 | 0.218 | 100.000 | 1.000 | 0.000 | 21 |
| 低销量收缩-常规Ridge混合 | no_positive_history | 0.173 | 0.232 | 363.418 | 1.000 | 2.912 | 21 |
| 低销量收缩指数平滑 | no_positive_history | 0.173 | 0.232 | 363.418 | 1.000 | 2.912 | 21 |
| 低销量收缩Ridge | no_positive_history | 0.357 | 0.423 | 748.926 | 1.000 | 7.604 | 21 |
| 完整综合Ridge | no_positive_history | 0.415 | 0.616 | 871.017 | 1.000 | 9.385 | 21 |
| 低销量指数平滑-常规Ridge混合 | very_sparse | 0.445 | 1.232 | 185.746 | 124.000 | 154.884 | 518 |
| 问题一简单指数平滑 | very_sparse | 0.445 | 1.232 | 185.746 | 124.000 | 154.884 | 518 |
| 完整综合Ridge | very_sparse | 0.490 | 1.143 | 204.751 | 124.000 | 193.802 | 518 |
| 低销量收缩-常规Ridge混合 | very_sparse | 0.556 | 1.176 | 232.366 | 124.000 | 253.557 | 518 |
| 低销量收缩指数平滑 | very_sparse | 0.556 | 1.176 | 232.366 | 124.000 | 253.557 | 518 |
| 低销量收缩Ridge | very_sparse | 0.587 | 1.157 | 245.023 | 124.000 | 273.016 | 518 |

## 5. 方法取舍

严格递推门店-商品粒度下，本检验最佳策略为 **低销量指数平滑-常规Ridge混合**，WAPE=75.85%。问题一简单指数平滑 WAPE=76.35%，完整综合 Ridge WAPE=76.68%。

可作为论文中的稳健性补充方案，但仍需说明其只是规则化后处理。

## 6. 误差显著性检验

| comparison | baseline_model | baseline_label | candidate_model | candidate_label | paired_unit | n_pairs | baseline_WAPE_pct | candidate_WAPE_pct | WAPE_diff_baseline_minus_candidate_pct_points | wilcoxon_stat | wilcoxon_p_value_greater | bootstrap_n | bootstrap_CI_lower_pct_points | bootstrap_CI_upper_pct_points | conclusion | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 综合 Ridge vs 问题一简单指数平滑 | q1_store_product_exp_smoothing | 问题一简单指数平滑 | ridge_full_external | 完整综合Ridge | window_start + store_id + product_id 的 7 日绝对误差 | 304 | 76.346 | 76.684 | -0.338 | 23332.000 | 0.461 | 5000 | -1.560 | 1.508 | 未达统计显著改进 | p<0.05 且 CI 不含 0 才判定误差显著下降 |
| 低销量混合策略 vs 问题一简单指数平滑 | q1_store_product_exp_smoothing | 问题一简单指数平滑 | hybrid_low_q1_regular_ridge | 低销量指数平滑-常规Ridge混合 | window_start + store_id + product_id 的 7 日绝对误差 | 304 | 76.346 | 75.853 | 0.494 | 9698.000 | 0.050 | 5000 | -0.719 | 1.335 | 未达统计显著改进 | p<0.05 且 CI 不含 0 才判定误差显著下降 |

检验以同一严格 7 日递推验证集为基础。Wilcoxon 检验的配对单位为“验证窗口 × 门店--商品序列”的 7 日绝对误差；bootstrap 置信区间按 7 日窗口块重采样计算 WAPE 差。若 p 值不小于 0.05 或置信区间包含 0，本文不写“误差显著改进”。

论文表述中将低销量分析写入模型评价和局限性部分：低销量序列是细粒度误差的主要来源之一；简单的层级收缩可作为稳健性检查，但若未显著优于主模型，就不应仅因模型形式更复杂而替换最终预测模型。
