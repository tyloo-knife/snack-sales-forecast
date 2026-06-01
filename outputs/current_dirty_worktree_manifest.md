# 当前脏工作区保护性清单

生成时间：2026-05-31

## 1. 当前分支

`codex/award-paper-iteration`

## 2. 最近 5 个 commit

```text
87f99c9 stage9: assemble submission package
c32026f stage8: generate final LaTeX paper PDF
cab47ed stage6: draft paper and final review materials
2347ddb Add q4 diagnostics and team onboarding
44daaad Add final method selection summary
```

## 3. 已修改文件列表

```text
 M .gitignore
 M README.md
 M RESULT_LOG.md
 M outputs/ai_usage_record.md
 M outputs/defense_notes.md
 M outputs/final_pdf_review.md
 M outputs/final_review_checklist.md
 M outputs/final_submission_review.md
 M outputs/method_search/final_method_selection.md
 M outputs/q4_error_diagnosis_report.md
 M paper/abstract.md
 M paper/appendix.md
 M paper/conclusion.md
 M paper/final_paper.pdf
 M paper/full_paper_draft.md
 M paper/latex/main.tex
 M paper/latex/sections/abstract.tex
 M paper/latex/sections/appendix.tex
 M paper/latex/sections/model_evaluation.tex
 M paper/latex/sections/q4_model.tex
 M paper/latex/sections/strengths_weaknesses.tex
 M paper/model_building.md
 M paper/model_evaluation.md
 M paper/model_solution.md
 M paper/result_analysis.md
 M submission/README_submission.md
 M submission/ai_usage_record.md
 M submission/final_7day_forecast.csv
 M submission/final_7day_forecast_integer.csv
 M submission/final_paper.pdf
 M submission/final_submission_review.md
```

## 4. 未跟踪文件列表

```text
?? figures/final_forecast_interval_by_store.png
?? figures/q4_ablation_store_product_wape.png
?? figures/q4_candidate_model_wape.png
?? figures/q4_hybrid_error_by_horizon.png
?? figures/q4_hybrid_forecast_7day_total_by_category.png
?? figures/q4_hybrid_forecast_7day_total_by_store.png
?? figures/q4_low_volume_strategy_wape.png
?? outputs/current_dirty_worktree_manifest.md
?? outputs/final_7day_forecast_hybrid_low_volume.csv
?? outputs/final_7day_forecast_hybrid_low_volume_integer.csv
?? outputs/final_forecast_interval_by_category.csv
?? outputs/final_forecast_interval_by_product.csv
?? outputs/final_forecast_interval_by_store.csv
?? outputs/final_freeze_audit_20260531.md
?? outputs/forecast_uncertainty_report.md
?? outputs/paper_format_benchmark_review.md
?? outputs/q4_ablation_metrics.csv
?? outputs/q4_ablation_predictions.csv
?? outputs/q4_ablation_report.md
?? outputs/q4_ablation_summary.json
?? outputs/q4_award_iteration_summary.json
?? outputs/q4_candidate_leave_one_window_selection.csv
?? outputs/q4_candidate_model_metrics.csv
?? outputs/q4_candidate_model_predictions.csv
?? outputs/q4_candidate_model_review.md
?? outputs/q4_candidate_window_metrics.csv
?? outputs/q4_error_attribution_enhanced_report.md
?? outputs/q4_hybrid_error_by_calendar_segment.csv
?? outputs/q4_hybrid_error_by_demand_class.csv
?? outputs/q4_hybrid_error_by_horizon.csv
?? outputs/q4_hybrid_final_forecast_report.md
?? outputs/q4_hybrid_final_forecast_summary.json
?? outputs/q4_hybrid_forecast_7day_total_by_category.csv
?? outputs/q4_hybrid_forecast_7day_total_by_product.csv
?? outputs/q4_hybrid_forecast_7day_total_by_store.csv
?? outputs/q4_hybrid_future_low_volume_classification.csv
?? outputs/q4_low_volume_classification.csv
?? outputs/q4_low_volume_classification_summary.csv
?? outputs/q4_low_volume_strategy_by_class.csv
?? outputs/q4_low_volume_strategy_metrics.csv
?? outputs/q4_low_volume_strategy_predictions.csv
?? outputs/q4_low_volume_strategy_report.md
?? outputs/q4_low_volume_strategy_summary.json
?? paper/cover_member_info.csv
?? paper/cover_member_info_template.md
?? paper/final_paper_award_iteration.pdf
?? src/fill_cover_members.py
?? src/stage5_q4_ablation.py
?? src/stage5_q4_award_iteration.py
?? src/stage5_q4_hybrid_final_forecast.py
?? src/stage5_q4_low_volume_strategy.py
?? submission/final_freeze_audit_20260531.md
?? submission/final_pdf_review.md
?? submission/final_review_checklist.md
?? tables/final_7day_forecast_hybrid_low_volume.csv
?? tables/final_7day_forecast_hybrid_low_volume_integer.csv
?? tables/final_forecast_interval_by_category.csv
?? tables/final_forecast_interval_by_product.csv
?? tables/final_forecast_interval_by_store.csv
?? tables/q4_ablation_metrics.csv
?? tables/q4_ablation_predictions.csv
?? tables/q4_candidate_leave_one_window_selection.csv
?? tables/q4_candidate_model_metrics.csv
?? tables/q4_candidate_model_predictions.csv
?? tables/q4_candidate_window_metrics.csv
?? tables/q4_hybrid_error_by_calendar_segment.csv
?? tables/q4_hybrid_error_by_demand_class.csv
?? tables/q4_hybrid_error_by_horizon.csv
?? tables/q4_hybrid_forecast_7day_total_by_category.csv
?? tables/q4_hybrid_forecast_7day_total_by_product.csv
?? tables/q4_hybrid_forecast_7day_total_by_store.csv
?? tables/q4_hybrid_future_low_volume_classification.csv
?? tables/q4_low_volume_classification.csv
?? tables/q4_low_volume_classification_summary.csv
?? tables/q4_low_volume_strategy_by_class.csv
?? tables/q4_low_volume_strategy_metrics.csv
?? tables/q4_low_volume_strategy_predictions.csv
```

## 5. 明显不应提交的临时文件

```text
paper\latex\compile_bibtex.log
paper\latex\compile_latexmk_20260531.log
paper\latex\compile_xelatex_1.log
paper\latex\compile_xelatex_2.log
paper\latex\compile_xelatex_3.log
paper\latex\main.aux
paper\latex\main.fdb_latexmk
paper\latex\main.fls
paper\latex\main.log
paper\latex\main.out
paper\latex\main.xdv
src\__pycache__
src\experimental\__pycache__
submission\compile_latexmk_20260531.log
```

说明：LaTeX 编译中间文件、Python 缓存、PDF 首页检查图片和本地 archive 备份不纳入 checkpoint commit。

## 6. 建议提交的文件或目录

```text
paper/
outputs/
tables/
figures/
src/
submission/
README.md
RESULT_LOG.md
TEAM_ONBOARDING.md
REVIEW_CHECKLIST.md
.gitignore
```

## 7. 建议暂不提交的文件或目录

```text
data/raw/
.venv/
__pycache__/
.pytest_cache/
.ipynb_checkpoints/
paper/latex/*.aux
paper/latex/*.log
paper/latex/*.out
paper/latex/*.toc
paper/latex/*.fls
paper/latex/*.fdb_latexmk
paper/latex/*.synctex.gz
paper/latex/*.xdv
paper/latex/main.pdf
submission/compile_*.log
outputs/final_paper_page*_check*.png
archive/
```

## 8. data/raw 修改检查

未发现 `data/raw/` 被修改。
