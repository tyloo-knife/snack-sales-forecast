# 提交包说明

本目录为数学建模 A 题《休闲零食连锁店商品销量预测》的提交前整理包。

## 文件说明

| 文件 | 说明 |
|---|---|
| `final_paper.pdf` | 正式论文 PDF。 |
| `final_7day_forecast_integer.csv` | 低销量混合策略整数化预测结果，适用于提交口径要求“销量件数”的情况。 |
| `final_7day_forecast.csv` | 低销量混合策略原始小数预测结果，保留为连续预测输出依据。 |
| `final_submission_review.md` | 提交前最终审查报告，列出材料状态、风险和人工确认事项。 |
| `final_review_checklist.md` | 最终审查清单。 |
| `final_freeze_audit_20260531.md` | 最终冻结审查记录，列出已完成项和赛前确认项。 |
| `final_pdf_review.md` | PDF 编译与口径检查报告。 |
| `compile_latexmk_20260531.log` | 最新 LaTeX 编译日志。 |
| `ai_usage_record.md` | AI 工具使用记录，是否提交取决于竞赛规则或学校要求。 |

## 注意事项

1. 原始数据未修改，且本提交包未复制 `data/raw/`。
2. 本提交包未包含 `.venv`、`__pycache__`、LaTeX 临时文件或中间渲染文件。
3. 若竞赛允许小数预测，可参考 `final_7day_forecast.csv`；若要求件数，优先使用 `final_7day_forecast_integer.csv`。两者均对应论文中的低销量指数平滑--常规 Ridge 混合策略。
4. 当前 PDF 封面成员信息已按用户提供内容填写；正式提交前仍需核对报名系统信息，若竞赛匿名提交则按匿名规则重新生成 PDF。
5. 正式提交前仍需人工确认竞赛平台要求的预测表格式和 AI 使用披露要求。
