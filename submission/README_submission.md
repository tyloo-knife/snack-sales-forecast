# 提交包说明

本目录为数学建模 A 题《休闲零食连锁店商品销量预测》的提交前整理包。

## 文件说明

| 文件 | 说明 |
|---|---|
| `final_paper.pdf` | 正式论文 PDF。 |
| `final_7day_forecast_integer.csv` | 整数化预测结果，适用于提交口径要求“销量件数”的情况。 |
| `final_7day_forecast.csv` | 模型原始小数预测结果，保留为连续预测输出依据。 |
| `final_submission_review.md` | 提交前最终审查报告，列出材料状态、风险和人工确认事项。 |
| `ai_usage_record.md` | AI 工具使用记录，是否提交取决于竞赛规则或学校要求。 |

## 注意事项

1. 原始数据未修改，且本提交包未复制 `data/raw/`。
2. 本提交包未包含 `.venv`、`__pycache__`、LaTeX 临时文件或中间渲染文件。
3. 若竞赛允许小数预测，可参考 `final_7day_forecast.csv`；若要求件数，优先使用 `final_7day_forecast_integer.csv`。
4. 正式提交前仍需人工确认竞赛平台要求的预测表格式和 AI 使用披露要求。
